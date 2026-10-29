import SwiftUI
import NetworkExtension
import URLGuardCore

/// Control panel for the on-device firewall.
@MainActor
final class FirewallController: ObservableObject {
    @Published var enabled = false
    @Published var status = "Not configured"
    @Published var error: String?
    @Published var threshold: Double = FirewallPolicy.defaultThreshold

    private var manager: NETunnelProviderManager?
    private let shared = UserDefaults(suiteName: "group.com.aics.urlguard")

    init() {
        threshold = shared?.double(forKey: "dns_threshold") ?? 0
        if threshold <= 0 { threshold = FirewallPolicy.defaultThreshold }
    }

    func load() async {
        do {
            let managers = try await NETunnelProviderManager
                .loadAllFromPreferences()
            manager = managers.first
            enabled = manager?.isEnabled ?? false
            status = describe(manager?.connection.status)
        } catch {
            self.error = error.localizedDescription
        }
    }

    func setEnabled(_ on: Bool) async {
        do {
            let m = manager ?? NETunnelProviderManager()
            let proto = (m.protocolConfiguration as? NETunnelProviderProtocol)
                ?? NETunnelProviderProtocol()
            proto.providerBundleIdentifier = "com.aics.urlguard.firewall"
            // Required by NetworkExtension, but nothing is tunnelled to it:
            // the provider answers locally and never opens a remote session.
            proto.serverAddress = "On-device"
            m.protocolConfiguration = proto
            m.localizedDescription = "URLGuard firewall"
            m.isEnabled = on
            try await m.saveToPreferences()
            try await m.loadFromPreferences()
            manager = m
            if on { try m.connection.startVPNTunnel() }
            else { m.connection.stopVPNTunnel() }
            enabled = on
            status = describe(m.connection.status)
        } catch {
            self.error = error.localizedDescription
            enabled = false
        }
    }

    func persistThreshold() {
        shared?.set(threshold, forKey: "dns_threshold")
        // Nudge the running extension to re-read its settings.
        if let session = manager?.connection as? NETunnelProviderSession {
            try? session.sendProviderMessage(Data()) { _ in }
        }
    }

    private func describe(_ s: NEVPNStatus?) -> String {
        switch s {
        case .connected: return "Running"
        case .connecting: return "Starting…"
        case .disconnecting: return "Stopping…"
        case .disconnected: return "Stopped"
        case .invalid, .none: return "Not configured"
        case .reasserting: return "Reconnecting…"
        @unknown default: return "Unknown"
        }
    }
}

struct FirewallView: View {
    @StateObject private var ctl = FirewallController()
    @State private var blocked: [BlockLogEntry] = []

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("On-device firewall", isOn: Binding(
                        get: { ctl.enabled },
                        set: { v in Task { await ctl.setEnabled(v) } }))
                    LabeledContent("Status") { Text(ctl.status) }
                } header: {
                    Text("DNS filtering")
                } footer: {
                    Text("Checks the names your device looks up against the "
                         + "on-device model and refuses the ones that score "
                         + "above the threshold.\n\nThis uses a local VPN "
                         + "profile because it is the only way iOS lets an app "
                         + "see DNS. Nothing is sent to a VPN server — there "
                         + "is no server. Queries are answered on the device "
                         + "and everything else is forwarded untouched.")
                }

                Section {
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Blocking threshold")
                            Spacer()
                            Text(String(format: "%.2f", ctl.threshold))
                                .monospacedDigit().foregroundStyle(.secondary)
                        }
                        Slider(value: $ctl.threshold, in: 0.50...0.99, step: 0.01)
                            .onChange(of: ctl.threshold) { _, _ in
                                ctl.persistThreshold()
                            }
                    }
                } footer: {
                    Text("Higher blocks less. The default of 0.95 was chosen "
                         + "because the model's measured false-positive rate "
                         + "at 0.65 is about 1 domain in 20 — enough to make "
                         + "the phone feel broken. At 0.95 it is about 1 in "
                         + "300, at roughly half the detection rate.")
                }

                if let e = ctl.error {
                    Section { Text(e).font(.caption).foregroundStyle(.red) }
                }

                Section("Recently blocked") {
                    if blocked.isEmpty {
                        Text("Nothing has been blocked.")
                            .foregroundStyle(.secondary).font(.callout)
                    } else {
                        ForEach(blocked, id: \.host) { b in
                            VStack(alignment: .leading, spacing: 2) {
                                Text(b.host).font(.callout)
                                Text(String(format: "scored %.2f · %@",
                                            b.score,
                                            b.at.formatted(.relative(presentation: .numeric))))
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                Section {
                    Text("Only blocked lookups are recorded, and only on this "
                         + "device. URLGuard does not keep a log of everywhere "
                         + "you go.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Firewall")
            .task {
                await ctl.load()
                blocked = BlockLogReader.shared.all()
            }
        }
    }
}

/// App-side view of the extension's block log.
struct BlockLogEntry: Codable {
    let host: String
    let score: Double
    let at: Date
}

enum BlockLogReader {
    static let shared = Reader()
    struct Reader {
        private let defaults = UserDefaults(suiteName: "group.com.aics.urlguard")
        func all() -> [BlockLogEntry] {
            guard let d = defaults?.data(forKey: "blocked_hosts"),
                  let items = try? JSONDecoder().decode([BlockLogEntry].self,
                                                        from: d)
            else { return [] }
            return items
        }
    }
}
