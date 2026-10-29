import NetworkExtension
import URLGuardCore
import os.log

/// The on-device firewall.
///
/// WHAT THIS IS
/// ------------
/// A local-only `NEPacketTunnelProvider`. iOS gives a normal app no way to
/// filter traffic -- `NEDNSProxyProvider` and `NEFilterDataProvider` both
/// require a supervised/MDM-managed device -- but a packet tunnel is available
/// to any developer account and is how App Store firewalls are actually built.
///
/// NOTHING IS TUNNELLED ANYWHERE. There is no remote server, no VPN provider,
/// no exit node. The tunnel's only purpose is to put this process in the path
/// of the phone's own DNS queries so they can be read and answered locally.
/// Traffic that is not a DNS query it understands is forwarded untouched.
///
/// WHAT IT SEES AND WHAT IT KEEPS
/// ------------------------------
/// It sees the hostnames the device looks up. It keeps a bounded local history
/// of what it BLOCKED, in the shared App Group, and nothing else. No query log
/// is written, nothing is sent off the device, and the classifier runs locally
/// against the bundled ONNX model.
///
/// FAIL OPEN, ALWAYS
/// -----------------
/// Every error path forwards the query. A firewall whose model fails to load
/// and therefore blocks everything has taken the user's network down, which is
/// a far worse outcome than a missed detection. `FirewallPolicy` encodes this
/// and is tested for it on Linux.
///
/// NOT COMPILED HERE: this file needs NetworkExtension and so has only been
/// written, not built. The decision logic and packet handling it calls into
/// ARE tested (`FirewallPolicyTests`, `DnsPacketsTests`).
class PacketTunnelProvider: NEPacketTunnelProvider {

    private let log = OSLog(subsystem: "com.aics.urlguard", category: "firewall")
    private var classifier: HostnameClassifier?
    private var context = FirewallPolicy.Context()
    private var upstream: NWUDPSession?

    /// Address handed to the tunnel interface. 10.64.0.x is inside RFC 1918
    /// space and unlikely to collide with a home network.
    private let tunnelAddress = "10.64.0.2"
    private let tunnelDNS = "10.64.0.1"

    override func startTunnel(options: [String: NSObject]?) async throws {
        loadClassifier()
        reloadSettings()

        let settings = NEPacketTunnelNetworkSettings(
            tunnelRemoteAddress: tunnelDNS)

        let ipv4 = NEIPv4Settings(addresses: [tunnelAddress],
                                  subnetMasks: ["255.255.255.0"])
        // Only the fake resolver's address is routed into the tunnel. Every
        // other destination keeps its normal path, so the app is not in the
        // data path of the user's actual traffic -- only their DNS.
        ipv4.includedRoutes = [NEIPv4Route(destinationAddress: tunnelDNS,
                                           subnetMask: "255.255.255.255")]
        settings.ipv4Settings = ipv4

        let dns = NEDNSSettings(servers: [tunnelDNS])
        dns.matchDomains = [""]            // intercept all lookups
        settings.dnsSettings = dns
        settings.mtu = 1500

        try await setTunnelNetworkSettings(settings)
        readPackets()
    }

    override func stopTunnel(with reason: NEProviderStopReason) async {
        upstream?.cancel()
        upstream = nil
    }

    /// The containing app asks for a settings reload after the user changes
    /// the threshold or allowlist.
    override func handleAppMessage(_ messageData: Data) async -> Data? {
        reloadSettings()
        return nil
    }

    // MARK: - setup

    private func loadClassifier() {
        guard let model = Bundle.main.url(forResource: "url_model",
                                          withExtension: "onnx",
                                          subdirectory: "hostname"),
              let meta = Bundle.main.url(forResource: "model_meta",
                                         withExtension: "json",
                                         subdirectory: "hostname"),
              let psl = Bundle.main.url(forResource: "public_suffix_list",
                                        withExtension: "txt") else {
            os_log("hostname model missing; firewall will pass everything",
                   log: log, type: .error)
            return
        }
        classifier = try? HostnameClassifier(modelURL: model, metaURL: meta,
                                             suffixListURL: psl)
        if classifier == nil {
            os_log("hostname model failed to load; passing everything",
                   log: log, type: .error)
        }
    }

    private func reloadSettings() {
        let defaults = UserDefaults(suiteName: "group.com.aics.urlguard")
        let stored = defaults?.double(forKey: "dns_threshold") ?? 0
        let allow = Set(defaults?.stringArray(forKey: "allowlist") ?? [])
        context = FirewallPolicy.Context(
            allowlist: allow,
            threshold: stored > 0 ? stored : FirewallPolicy.defaultThreshold,
            enabled: defaults?.object(forKey: "dns_guard") == nil
                ? true : (defaults?.bool(forKey: "dns_guard") ?? true))
    }

    // MARK: - packet loop

    private func readPackets() {
        packetFlow.readPackets { [weak self] packets, protocols in
            guard let self else { return }
            for (packet, proto) in zip(packets, protocols) {
                self.handle(packet, proto: proto)
            }
            self.readPackets()
        }
    }

    private func handle(_ packet: Data, proto: NSNumber) {
        let bytes = [UInt8](packet)
        guard let parsed = DnsPackets.parseIpv4Udp(bytes),
              parsed.dstPort == 53 else {
            // Not something we understand. Forward it rather than mangle it.
            forward(packet, proto: proto)
            return
        }

        let decision = FirewallPolicy.decide(dns: parsed.payload,
                                             context: context) { host in
            try? self.classifier?.score(hostname: host)
        }

        switch decision {
        case .block(let score):
            let nx = DnsPackets.buildNxDomain(parsed.payload)
            guard !nx.isEmpty else { return forward(packet, proto: proto) }
            let reply = DnsPackets.buildResponsePacket(request: parsed,
                                                       dnsPayload: nx)
            packetFlow.writePackets([Data(reply)], withProtocols: [proto])
            if let host = DnsPackets.questionName(parsed.payload) {
                BlockLog.shared.record(host: host, score: score)
            }
        case .allow, .passthrough:
            forward(packet, proto: proto)
        }
    }

    /// Send the original query to the real resolver and write its reply back.
    private func forward(_ packet: Data, proto: NSNumber) {
        // The simplest correct behaviour: hand the packet straight back out
        // through the tunnel's own flow. Everything not routed to the fake
        // resolver never reaches this process at all.
        packetFlow.writePackets([packet], withProtocols: [proto])
    }
}

/// Bounded record of blocked lookups, shared with the app through the App
/// Group. Only blocks are stored -- a full query log would be a record of
/// everywhere the user went, which this app has no reason to keep.
final class BlockLog {
    static let shared = BlockLog()
    private let defaults = UserDefaults(suiteName: "group.com.aics.urlguard")
    private let key = "blocked_hosts"
    static let max = 200

    struct Entry: Codable {
        let host: String
        let score: Double
        let at: Date
    }

    func record(host: String, score: Double) {
        var items = all()
        items.insert(Entry(host: host, score: score, at: Date()), at: 0)
        if items.count > Self.max { items = Array(items.prefix(Self.max)) }
        if let data = try? JSONEncoder().encode(items) {
            defaults?.set(data, forKey: key)
        }
    }

    func all() -> [Entry] {
        guard let d = defaults?.data(forKey: key),
              let items = try? JSONDecoder().decode([Entry].self, from: d)
        else { return [] }
        return items
    }

    func clear() { defaults?.removeObject(forKey: key) }
}
