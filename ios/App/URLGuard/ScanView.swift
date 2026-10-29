import SwiftUI
import URLGuardCore

/// The scanner. Mirrors the web UI's conservative presentation: a band, not a
/// verdict; confidence as distance from the threshold; signals only for
/// measurements actually present.
struct ScanView: View {
    @EnvironmentObject var model: AppModel
    @State private var input = ""
    @State private var result: RiskAssessment.Result?
    @State private var scannedURL = ""
    @State private var elapsedMS: Double?
    @State private var error: String?
    @State private var busy = false
    @State private var serverBand: String?
    @State private var serverNote: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("https://example.com/path", text: $input)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    Button(busy ? "Analysing…" : "Analyse") { scan() }
                        .disabled(input.trimmingCharacters(
                            in: .whitespaces).isEmpty || busy)
                } header: {
                    Text("URL to analyse")
                } footer: {
                    Text("The URL is analysed as text on this device. It is "
                         + "never fetched, resolved or opened.")
                }

                if let error {
                    Section { Text(error).foregroundStyle(.red) }
                }

                if let r = result {
                    Section("Assessment") {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(r.band.label).font(.title2.bold())
                                    .foregroundStyle(colour(r.band))
                                Text(r.band.detail).font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            VStack(alignment: .trailing) {
                                Text("\(r.score)").font(.largeTitle.bold())
                                    .monospacedDigit()
                                Text("risk / 100").font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        LabeledContent("Confidence") {
                            Text(String(format: "%.1f%% · %@", r.confidence,
                                        r.confidenceLabel)).monospacedDigit()
                        }
                        if let elapsedMS {
                            LabeledContent("Detection time") {
                                Text(String(format: "%.1f ms", elapsedMS))
                                    .monospacedDigit()
                            }
                        }
                        LabeledContent("Model") {
                            Text(model.classifier?.modelVersion ?? "—")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }

                    if !r.signals.isEmpty {
                        Section("Security signals") {
                            ForEach(r.signals, id: \.self) { Text($0).font(.callout) }
                        }
                    }

                    if let serverBand {
                        Section("Server analysis") {
                            LabeledContent("Band") { Text(serverBand) }
                            if let serverNote {
                                Text(serverNote).font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    Section {
                        Text(RiskAssessment.disclaimer).font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("URLGuard")
            .onChange(of: model.pendingURL) { _, new in
                guard !new.isEmpty else { return }
                input = new
                model.pendingURL = ""
                scan()
            }
        }
    }

    private func colour(_ b: RiskAssessment.Band) -> Color {
        switch b {
        case .low: return .green
        case .suspicious: return .orange
        case .high: return .red
        case .insufficient: return .gray
        }
    }

    private func scan() {
        guard let clf = model.classifier else {
            error = model.loadError ?? "The on-device model is unavailable."
            return
        }
        error = nil; busy = true; serverBand = nil; serverNote = nil
        let target = input.trimmingCharacters(in: .whitespaces)
        let started = Date()
        do {
            let verdict = try clf.classify(target)
            elapsedMS = Date().timeIntervalSince(started) * 1000
            result = RiskAssessment.make(proba: verdict.score,
                                         threshold: verdict.threshold,
                                         feats: verdict.features)
            scannedURL = target
            ScanHistory.shared.add(url: target, score: result!.score,
                                   band: result!.band.rawValue)
            busy = false

            // Only a borderline score is worth a server round trip, and only
            // if the user turned it on. Everything else is already decided.
            let margin = RiskAssessment.decisionMargin(verdict.score,
                                                       threshold: verdict.threshold)
            if model.deepScanEnabled && margin <= 0.20 {
                Task { await deepScan(target) }
            }
        } catch {
            busy = false
            self.error = "Analysis failed: \(error.localizedDescription)"
        }
    }

    private func deepScan(_ target: String) async {
        do {
            let json = try await ApiClient.shared.scan(url: target)
            guard let a = json["assessment"] as? [String: Any],
                  let band = a["risk_band"] as? [String: Any] else { return }
            await MainActor.run {
                serverBand = band["label"] as? String
                serverNote = "The server re-analysed this URL with its full "
                    + "layer stack. The on-device result above is unchanged."
            }
        } catch {
            await MainActor.run {
                serverNote = "The server could not be reached; the on-device "
                    + "result stands."
            }
        }
    }
}
