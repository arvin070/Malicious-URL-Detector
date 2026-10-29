import UIKit
import Social
import UniformTypeIdentifiers
import URLGuardCore

/// "Share → URLGuard" from any app.
///
/// This is the closest iOS allows to Android's link interception. Android can
/// register as a link handler and sit between a tap and the browser; iOS gives
/// a normal app no such hook, so the user has to hand the link over
/// deliberately. That is a real capability gap, not a design choice, and the
/// UI says so rather than implying full coverage.
///
/// Scoring happens here, in the extension process, with no network call: the
/// same ONNX model and the same feature extractor the main app uses.
class ShareViewController: UIViewController {

    private let label = UILabel()
    private let detail = UILabel()
    private let scoreLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setUpViews()
        extractURL { [weak self] url in
            guard let self else { return }
            guard let url else {
                self.show(title: "No link found",
                          detail: "The shared item did not contain a URL.",
                          score: nil, colour: .secondaryLabel)
                return
            }
            self.assess(url)
        }
    }

    private func setUpViews() {
        [label, detail, scoreLabel].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.numberOfLines = 0
            view.addSubview($0)
        }
        label.font = .preferredFont(forTextStyle: .title1)
        scoreLabel.font = .monospacedDigitSystemFont(ofSize: 44, weight: .bold)
        detail.font = .preferredFont(forTextStyle: .callout)
        detail.textColor = .secondaryLabel

        let done = UIButton(type: .system)
        done.setTitle("Done", for: .normal)
        done.addTarget(self, action: #selector(finish), for: .touchUpInside)
        done.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(done)

        NSLayoutConstraint.activate([
            scoreLabel.topAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 32),
            scoreLabel.leadingAnchor.constraint(
                equalTo: view.leadingAnchor, constant: 24),
            label.topAnchor.constraint(equalTo: scoreLabel.bottomAnchor, constant: 4),
            label.leadingAnchor.constraint(equalTo: scoreLabel.leadingAnchor),
            label.trailingAnchor.constraint(
                equalTo: view.trailingAnchor, constant: -24),
            detail.topAnchor.constraint(equalTo: label.bottomAnchor, constant: 12),
            detail.leadingAnchor.constraint(equalTo: label.leadingAnchor),
            detail.trailingAnchor.constraint(equalTo: label.trailingAnchor),
            done.topAnchor.constraint(equalTo: detail.bottomAnchor, constant: 28),
            done.leadingAnchor.constraint(equalTo: label.leadingAnchor),
        ])
    }

    private func extractURL(_ completion: @escaping (String?) -> Void) {
        guard let item = extensionContext?.inputItems.first as? NSExtensionItem,
              let providers = item.attachments else {
            return completion(nil)
        }
        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
                provider.loadItem(forTypeIdentifier: UTType.url.identifier) { v, _ in
                    DispatchQueue.main.async {
                        completion((v as? URL)?.absoluteString)
                    }
                }
                return
            }
        }
        for provider in providers
        where provider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) {
            provider.loadItem(forTypeIdentifier: UTType.plainText.identifier) { v, _ in
                DispatchQueue.main.async { completion(v as? String) }
            }
            return
        }
        completion(nil)
    }

    private func assess(_ url: String) {
        guard let modelURL = Bundle.main.url(forResource: "url_model",
                                             withExtension: "onnx"),
              let metaURL = Bundle.main.url(forResource: "model_meta",
                                            withExtension: "json"),
              let pslURL = Bundle.main.url(forResource: "public_suffix_list",
                                           withExtension: "txt"),
              let clf = try? UrlClassifier(modelURL: modelURL, metaURL: metaURL,
                                           suffixListURL: pslURL),
              let verdict = try? clf.classify(url) else {
            // Fail open, as everywhere else in URLGuard: say the check could
            // not run rather than implying the link is fine.
            show(title: "Check unavailable",
                 detail: "The on-device model could not be loaded, so this "
                       + "link was not assessed.",
                 score: nil, colour: .secondaryLabel)
            return
        }

        let r = RiskAssessment.make(proba: verdict.score,
                                    threshold: verdict.threshold,
                                    feats: verdict.features)
        ScanHistory.shared.add(url: url, score: r.score, band: r.band.rawValue)

        let colour: UIColor
        switch r.band {
        case .low: colour = .systemGreen
        case .suspicious: colour = .systemOrange
        case .high: colour = .systemRed
        case .insufficient: colour = .systemGray
        }
        var text = r.band.detail
        if !r.signals.isEmpty {
            text += "\n\n" + r.signals.joined(separator: "\n")
        }
        text += "\n\n" + RiskAssessment.disclaimer
        show(title: r.band.label, detail: text, score: r.score, colour: colour)
    }

    private func show(title: String, detail: String, score: Int?,
                      colour: UIColor) {
        label.text = title
        label.textColor = colour
        scoreLabel.text = score.map { "\($0)" } ?? "—"
        scoreLabel.textColor = colour
        self.detail.text = detail
    }

    @objc private func finish() {
        extensionContext?.completeRequest(returningItems: nil)
    }
}
