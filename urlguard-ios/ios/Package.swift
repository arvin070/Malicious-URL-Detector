// swift-tools-version: 5.9
import PackageDescription

// URLGuardCore is deliberately a plain SwiftPM library with no Apple-only
// dependencies, so the feature port can be COMPILED AND TESTED ON LINUX
// against the same golden vectors the Kotlin port is held to. The iOS app,
// Safari extension and share extension live outside this package (see
// ios/App/) because they need UIKit/SwiftUI and an Xcode build.
//
// That split is the point: the part that has to agree with Python to the last
// decimal is the part that can be verified without a Mac.
let package = Package(
    name: "URLGuardCore",
    platforms: [.iOS(.v15), .macOS(.v12)],
    products: [
        .library(name: "URLGuardCore", targets: ["URLGuardCore"]),
    ],
    targets: [
        .target(name: "URLGuardCore"),
        .testTarget(
            name: "URLGuardCoreTests",
            dependencies: ["URLGuardCore"]
        ),
    ]
)
