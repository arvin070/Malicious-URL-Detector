import Foundation
import URLGuardCore
import onnxruntime_objc

/// On-device scoring with the same ONNX model the Android app runs.
///
/// NOT COMPILED OR RUN ON LINUX
/// ----------------------------
/// Everything in ios/App depends on Apple frameworks, so unlike
/// `URLGuardCore` it cannot be built or tested from this repo's Linux
/// toolchain. The part that has to agree with Python to the last decimal --
/// feature extraction -- deliberately lives in URLGuardCore and IS verified
/// there against the golden vectors. What remains here is model plumbing.
///
/// The model file and its metadata are the SAME artefacts the Android build
/// ships (`android/app/src/main/assets/`), referenced rather than copied, so
/// the two platforms cannot end up on different model versions.
public final class UrlClassifier {

    public struct Verdict {
        public let url: String
        public let score: Double
        public let malicious: Bool
        public let threshold: Double
        public let features: [String: Double]
    }

    private let session: ORTSession
    private let env: ORTEnv
    private let featureNames: [String]
    private let tld: TldExtract
    public let threshold: Double
    public let modelVersion: String

    private struct Meta: Decodable {
        let threshold: Double
        let feature_names: [String]
        let model_version: String?
    }

    public init(modelURL: URL, metaURL: URL, suffixListURL: URL) throws {
        let meta = try JSONDecoder().decode(Meta.self,
                                            from: Data(contentsOf: metaURL))
        self.featureNames = meta.feature_names
        self.threshold = meta.threshold
        self.modelVersion = meta.model_version ?? "unknown"
        self.tld = try TldExtract.fromSuffixFile(suffixListURL)

        self.env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        // Single-threaded on purpose: the model is tiny, and thread pools on
        // a phone cost more in wakeups than they save in latency.
        try options.setIntraOpNumThreads(1)
        self.session = try ORTSession(env: env,
                                      modelPath: modelURL.path,
                                      sessionOptions: options)
    }

    /// Score one URL. Pure local computation -- nothing is fetched or resolved.
    public func classify(_ url: String) throws -> Verdict {
        let pairs = UrlFeatures.extract(url, tld: tld)
        var byName: [String: Double] = [:]
        byName.reserveCapacity(pairs.count)
        for (k, v) in pairs { byName[k] = v }

        var row = [Float](repeating: 0, count: featureNames.count)
        for (i, name) in featureNames.enumerated() {
            row[i] = Float(byName[name] ?? 0)
        }

        let data = NSMutableData(bytes: row, length: row.count * MemoryLayout<Float>.size)
        let input = try ORTValue(
            tensorData: data,
            elementType: .float,
            shape: [1, NSNumber(value: featureNames.count)])

        let inputName = try session.inputNames().first ?? "input"
        let outputs = try session.run(
            withInputs: [inputName: input],
            outputNames: Set(try session.outputNames()),
            runOptions: nil)

        let proba = try Self.probability(from: outputs,
                                         names: try session.outputNames())
        return Verdict(url: url, score: proba, malicious: proba >= threshold,
                       threshold: threshold, features: byName)
    }

    /// Pull P(malicious) out of whichever output the converter produced.
    ///
    /// skl2onnx/onnxmltools emit either a `probabilities` tensor or a
    /// sequence-of-maps, depending on the converter and opset. Both shapes are
    /// handled rather than assuming one, because guessing wrong yields a
    /// plausible number from the wrong column.
    private static func probability(from outputs: [String: ORTValue],
                                    names: [String]) throws -> Double {
        let preferred = names.first { $0.lowercased().contains("probab") }
            ?? names.last
        guard let name = preferred, let value = outputs[name] else {
            throw NSError(domain: "URLGuard", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "model produced no probability output",
            ])
        }
        let info = try value.tensorTypeAndShapeInfo()
        let raw = try value.tensorData() as Data
        let floats = raw.withUnsafeBytes { buf -> [Float] in
            Array(buf.bindMemory(to: Float.self))
        }
        guard !floats.isEmpty else {
            throw NSError(domain: "URLGuard", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "empty probability tensor",
            ])
        }
        // Binary classifier: [P(benign), P(malicious)]. A single-element
        // output is already P(malicious).
        let shape = info.shape.map(\.intValue)
        if floats.count >= 2 && (shape.last ?? 2) >= 2 {
            return Double(floats[1])
        }
        return Double(floats[0])
    }
}
