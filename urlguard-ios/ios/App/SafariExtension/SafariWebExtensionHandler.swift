import SafariServices
import os.log
import URLGuardCore

/// Native half of the Safari Web Extension.
///
/// The JavaScript side cannot run ONNX, and shipping a second scoring
/// implementation in JS would be a third place for the features to drift from
/// Python. So the content script sends the URL here, and this scores it with
/// the same model and the same `URLGuardCore` extractor the app uses.
///
/// Nothing is fetched. The URL arrives as a string and is treated as one.
class SafariWebExtensionHandler: NSObject, NSExtensionRequestHandling {

    private static let classifier: UrlClassifier? = {
        guard let m = Bundle.main.url(forResource: "url_model", withExtension: "onnx"),
              let meta = Bundle.main.url(forResource: "model_meta", withExtension: "json"),
              let psl = Bundle.main.url(forResource: "public_suffix_list",
                                        withExtension: "txt")
        else { return nil }
        return try? UrlClassifier(modelURL: m, metaURL: meta, suffixListURL: psl)
    }()

    func beginRequest(with context: NSExtensionContext) {
        let item = context.inputItems.first as? NSExtensionItem
        let message = item?.userInfo?[SFExtensionMessageKey] as? [String: Any]
        let url = message?["url"] as? String ?? ""

        let response = NSExtensionItem()
        response.userInfo = [SFExtensionMessageKey: assess(url)]
        context.completeRequest(returningItems: [response])
    }

    private func assess(_ url: String) -> [String: Any] {
        guard !url.isEmpty else {
            return ["available": false, "reason": "no url supplied"]
        }
        guard let clf = Self.classifier, let v = try? clf.classify(url) else {
            // Fail open: Safari must not be blocked because the model failed
            // to load.
            return ["available": false,
                    "reason": "the on-device model is unavailable"]
        }
        let r = RiskAssessment.make(proba: v.score, threshold: v.threshold,
                                    feats: v.features)
        ScanHistory.shared.add(url: url, score: r.score, band: r.band.rawValue)
        return [
            "available": true,
            "score": r.score,
            "band": r.band.rawValue,
            "label": r.band.label,
            "detail": r.band.detail,
            "signals": r.signals,
            "disclaimer": RiskAssessment.disclaimer,
            "model": clf.modelVersion,
        ]
    }
}
