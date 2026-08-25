// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "FreeTokenApp",
    platforms: [
        .macOS(.v13)
    ],
    targets: [
        .executableTarget(
            name: "FreeTokenApp",
            path: "Sources/FreeTokenApp"
        ),
        .testTarget(
            name: "FreeTokenAppTests",
            dependencies: ["FreeTokenApp"],
            path: "Tests/FreeTokenAppTests"
        ),
    ]
)
