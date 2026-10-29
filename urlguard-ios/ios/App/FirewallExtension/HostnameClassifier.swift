import Foundation
import URLGuardCore
import onnxruntime_objc

/// Scores a bare hostname with the hostname-scope model.
///
/// A SEPARATE MODEL ON PURPOSE
/// ---------------------------
/// The DNS layer never sees a URL -- only a name. Feeding a bare hostname to
/// the full-URL model would hand it input unlike anything in its training
/// data: no scheme, no path, no query, so every path- and query-derived
/// feature reads zero and the score is meaningless. `models/v3_hostname` is
/// trained on hostnames alone, which is why it exists and why the Android app
/// ships it alongside the URL model.
///
/// Not compiled here (needs onnxruntime-objc); the feature extraction it
/// relies on is tested in URLGuardCore.
final class HostnameClassifier {

    private let session: ORTSession
    private let env: ORTEnv
    private let featureNames: [String]
    private let tld: TldExtract
    let threshold: Double

    private struct Meta: Decodable {
        let threshold: Double
        let feature_names: [String]
    }

    init(modelURL: URL, metaURL: URL, suffixListURL: URL) throws {
        let meta = try JSONDecoder().decode(Meta.self,
                                            from: Data(contentsOf: metaURL))
        featureNames = meta.feature_names
        threshold = meta.threshold
        tld = try TldExtract.fromSuffixFile(suffixListURL)
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(1)
        session = try ORTSession(env: env, modelPath: modelURL.path,
                                 sessionOptions: opts)
    }

    /// P(malicious) for one hostname.
    ///
    /// The hostname is passed through exactly as the Python hostname pipeline
    /// does -- as a bare name, with nothing prepended. An earlier version of
    /// the dataset builder synthesised "http://{domain}/" around bare domains
    /// and gave `has_path` an AUC of 0.92 against the label: the model learned
    /// the synthesis, not the phishing. Do not add a scheme here.
    func score(hostname: String) throws -> Double {
        let values = UrlFeatures.vector(hostname, tld: tld,
                                        featureNames: featureNames)
        var row = values.map { Float($0) }
        let data = NSMutableData(bytes: &row,
                                 length: row.count * MemoryLayout<Float>.size)
        let input = try ORTValue(tensorData: data, elementType: .float,
                                 shape: [1, NSNumber(value: row.count)])
        let inputName = try session.inputNames().first ?? "input"
        let outputs = try session.run(withInputs: [inputName: input],
                                      outputNames: Set(try session.outputNames()),
                                      runOptions: nil)
        let names = try session.outputNames()
        let preferred = names.first { $0.lowercased().contains("probab") } ?? names.last
        guard let n = preferred, let v = outputs[n] else { return 0 }
        let info = try v.tensorTypeAndShapeInfo()
        let raw = try v.tensorData() as Data
        let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard !floats.isEmpty else { return 0 }
        let shape = info.shape.map(\.intValue)
        return floats.count >= 2 && (shape.last ?? 2) >= 2
            ? Double(floats[1]) : Double(floats[0])
    }
}
