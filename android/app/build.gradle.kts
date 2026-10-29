plugins {
    id("com.android.application")
}

android {
    namespace = "com.aics.urlguard"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.aics.urlguard"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // Ship one APK per real-device ABI instead of one universal 142 MB file.
    // ONNX Runtime's native library is ~23-39 MB PER ABI and dominates the
    // build; x86/x86_64 exist only for emulators.
    splits {
        abi {
            isEnable = true
            reset()
            // x86_64 is for the emulator only; phones use the arm builds.
            include("arm64-v8a", "armeabi-v7a", "x86_64")
            isUniversalApk = false
        }
    }

    signingConfigs {
        create("release") {
            // Throwaway local key, committed on purpose so the expo build is
            // reproducible on any machine. It is NOT a publishing key: if this
            // ever goes to Play, generate a fresh keystore and keep it out of
            // the repo.
            storeFile = file("urlguard-demo.keystore")
            storePassword = "urlguard"
            keyAlias = "urlguard"
            keyPassword = "urlguard"
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"),
                          "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
        getByName("test").java.srcDirs("src/test/kotlin")
    }

    // The ONNX model and the golden vectors must not be compressed: the model
    // is mmap'd by ONNX Runtime, and a compressed asset would have to be fully
    // inflated into RAM on every cold start.
    androidResources {
        noCompress += listOf("onnx")
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
        unitTests.all { it.useJUnitPlatform() }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-service:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")

    // Desktop ONNX Runtime so the full pipeline (features -> model -> score)
    // can be verified on the JVM against Python's own outputs. The Android AAR
    // ships Android-only natives and cannot run in a unit test.
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.30.0")
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
