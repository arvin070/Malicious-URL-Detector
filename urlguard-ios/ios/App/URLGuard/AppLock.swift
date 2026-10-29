import LocalAuthentication
import SwiftUI

/// Optional Face ID / Touch ID gate on the app.
///
/// WHAT THIS DOES AND DOES NOT PROTECT
/// -----------------------------------
/// It keeps someone holding an unlocked phone from reading the scan history
/// and the signed-in session. That is a real threat for this app, because the
/// history is a list of links the user has visited or been sent.
///
/// It is NOT a cryptographic control. The data is protected by iOS file
/// protection and the Keychain, not by this gate; a biometric prompt is a UI
/// check and anyone with the device passcode can get past it. It is presented
/// as what it is rather than as encryption.
///
/// Deliberately NOT included: jailbreak detection. Every published check is
/// bypassed by the tooling it tries to detect, and shipping one mainly
/// produces a false sense of assurance plus false positives on ordinary
/// devices.
@MainActor
final class AppLock: ObservableObject {
    @Published private(set) var locked: Bool
    @Published private(set) var unavailableReason: String?

    static let enabledKey = "app_lock"

    var isEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: Self.enabledKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: Self.enabledKey)
            locked = newValue
        }
    }

    init() {
        locked = UserDefaults.standard.bool(forKey: Self.enabledKey)
    }

    /// True when the device can actually do this, so the toggle is not offered
    /// where it would never work.
    func availability() -> String? {
        var error: NSError?
        let ok = LAContext().canEvaluatePolicy(
            .deviceOwnerAuthentication, error: &error)
        return ok ? nil : (error?.localizedDescription
                           ?? "This device cannot verify your identity.")
    }

    func unlock() async {
        guard locked else { return }
        let context = LAContext()
        context.localizedFallbackTitle = "Use passcode"
        do {
            // deviceOwnerAuthentication, not ...WithBiometrics: falling back
            // to the passcode is correct here. Refusing a passcode would lock
            // the user out of their own history whenever Face ID fails.
            let ok = try await context.evaluatePolicy(
                .deviceOwnerAuthentication,
                localizedReason: "Unlock URLGuard to see your scan history.")
            if ok { locked = false }
        } catch {
            unavailableReason = error.localizedDescription
        }
    }

    func lock() { if isEnabled { locked = true } }
}

/// Covers the UI while locked. A plain overlay, so nothing sensitive is in the
/// app switcher snapshot either.
struct LockScreen: View {
    @EnvironmentObject var lock: AppLock

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "lock.shield")
                .font(.system(size: 54)).foregroundStyle(.secondary)
            Text("URLGuard is locked").font(.title3.bold())
            Text("Your scan history is hidden until you unlock.")
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Unlock") { Task { await lock.unlock() } }
                .buttonStyle(.borderedProminent)
            if let r = lock.unavailableReason {
                Text(r).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.regularMaterial)
    }
}
