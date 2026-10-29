import SwiftUI
import URLGuardCore

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var lock: AppLock
    @State private var lockOn = UserDefaults.standard.bool(forKey: AppLock.enabledKey)
    @State private var email = ""
    @State private var password = ""
    @State private var status: String?
    @State private var signedIn = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("https://your-server:5000", text: $model.serverURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                        .onSubmit { model.persist() }
                } header: {
                    Text("Server")
                } footer: {
                    Text("Only needed for account history and deep scans. "
                         + "On-device scanning works without it.")
                }

                Section {
                    Toggle("Deep scan borderline URLs", isOn: $model.deepScanEnabled)
                        .onChange(of: model.deepScanEnabled) { _, _ in model.persist() }
                } footer: {
                    Text("When the on-device model is close to its decision "
                         + "threshold, ask the server for a second opinion. "
                         + "This transmits the URL being checked to your "
                         + "server, so it is off by default.")
                }

                if !signedIn {
                    Section("Sign in") {
                        TextField("Email", text: $email)
                            .textInputAutocapitalization(.never)
                            .keyboardType(.emailAddress)
                        SecureField("Password", text: $password)
                        Button("Sign in") { signIn() }
                            .disabled(email.isEmpty || password.isEmpty)
                    }
                } else {
                    Section {
                        Button("Sign out", role: .destructive) {
                            Task { await ApiClient.shared.logout()
                                   signedIn = false }
                        }
                    }
                }

                if let status {
                    Section { Text(status).font(.caption) }
                }

                Section {
                    if let why = lock.availability() {
                        Text(why).font(.caption).foregroundStyle(.secondary)
                    } else {
                        Toggle("Require Face ID / passcode", isOn: $lockOn)
                            .onChange(of: lockOn) { _, v in lock.isEnabled = v }
                    }
                } header: {
                    Text("App lock")
                } footer: {
                    Text("Hides your scan history behind the device's own "
                         + "authentication. This is a screen lock, not "
                         + "encryption — anyone with your passcode can pass "
                         + "it. Your tokens are stored in the Keychain either "
                         + "way.")
                }

                Section("Model") {
                    LabeledContent("On-device model") {
                        Text(model.classifier?.modelVersion ?? "not loaded")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let e = model.loadError {
                        Text(e).font(.caption).foregroundStyle(.red)
                    }
                }

                Section {
                    Text("URLGuard analyses URLs on this device. It does not "
                         + "resolve, fetch or open them. iOS does not allow an "
                         + "app to intercept links system-wide, so warnings in "
                         + "Safari come from the Safari extension and anywhere "
                         + "else from Share → URLGuard.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .task { signedIn = await ApiClient.shared.isSignedIn }
        }
    }

    private func signIn() {
        model.persist()
        Task {
            do {
                _ = try await ApiClient.shared.login(email: email,
                                                     password: password)
                signedIn = true
                status = "Signed in."
                password = ""
            } catch {
                status = "Sign-in failed: \(error.localizedDescription)"
            }
        }
    }
}
