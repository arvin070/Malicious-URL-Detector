import SwiftUI
import URLGuardCore

@main
struct URLGuardApp: App {
    @StateObject private var model = AppModel()
    @StateObject private var lock = AppLock()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            ZStack {
                RootView()
                    .environmentObject(model)
                    .environmentObject(lock)
                    // A link shared into the app, or opened via the
                    // urlguard:// scheme, goes straight to the scanner.
                    .onOpenURL { model.scanRequested($0) }
                if lock.locked {
                    LockScreen().environmentObject(lock)
                        .transition(.opacity)
                }
            }
            .task { await lock.unlock() }
            .onChange(of: phase) { _, new in
                // Re-lock as soon as the app leaves the foreground, so the
                // history is not readable from the app switcher.
                if new != .active { lock.lock() }
            }
        }
    }
}

/// Loads the classifier once and holds app-wide state.
@MainActor
final class AppModel: ObservableObject {
    @Published var classifier: UrlClassifier?
    @Published var loadError: String?
    @Published var pendingURL: String = ""
    @Published var serverURL: String = UserDefaults.standard
        .string(forKey: "server_url") ?? ""
    /// Off by default, like the Android app: a deep scan transmits the URL
    /// being checked to the server, which is the user's call to make.
    @Published var deepScanEnabled: Bool = UserDefaults.standard
        .bool(forKey: "deep_scan")

    init() { load() }

    func load() {
        do {
            guard let model = Bundle.main.url(forResource: "url_model",
                                              withExtension: "onnx"),
                  let meta = Bundle.main.url(forResource: "model_meta",
                                             withExtension: "json"),
                  let psl = Bundle.main.url(forResource: "public_suffix_list",
                                            withExtension: "txt") else {
                loadError = "The model files are missing from the app bundle."
                return
            }
            classifier = try UrlClassifier(modelURL: model, metaURL: meta,
                                           suffixListURL: psl)
        } catch {
            // Fail open: the app still works as a front end to the server.
            loadError = "The on-device model could not be loaded: "
                + error.localizedDescription
        }
    }

    func scanRequested(_ url: URL) {
        // urlguard://scan?url=... or a plain shared link.
        if url.scheme == "urlguard",
           let target = URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .queryItems?.first(where: { $0.name == "url" })?.value {
            pendingURL = target
        } else {
            pendingURL = url.absoluteString
        }
    }

    func persist() {
        UserDefaults.standard.set(serverURL, forKey: "server_url")
        UserDefaults.standard.set(deepScanEnabled, forKey: "deep_scan")
        if let u = URL(string: serverURL), u.scheme != nil {
            Task { await ApiClient.shared.configure(baseURL: u) }
        }
    }
}

struct RootView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        TabView {
            ScanView().tabItem { Label("Scan", systemImage: "magnifyingglass") }
            FirewallView().tabItem {
                Label("Firewall", systemImage: "shield.lefthalf.filled")
            }
            HistoryView().tabItem { Label("History", systemImage: "clock") }
            SettingsView().tabItem { Label("Settings", systemImage: "gear") }
        }
    }
}
