plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.vox.narrator"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
        isCoreLibraryDesugaringEnabled = true
    }

    defaultConfig {
        applicationId = "com.vox.narrator"
        // Foreground-service types and the on-device speech runtime need API 26+.
        minSdk = 26
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
        // sherpa-onnx ships 64-bit ARM builds for phones; CI's emulator test
        // builds for x86_64 instead (NARRATOR_ABIS=x86_64).
        ndk { abiFilters += (System.getenv("NARRATOR_ABIS") ?: "arm64-v8a").split(",") }
    }

    // Every release must be signed with the same key, or Android refuses to
    // install it over the previous one. CI decodes the key from the
    // VOX_KEYSTORE_B64 secret; local builds without it use the debug key.
    val releaseKeystore = System.getenv("VOX_KEYSTORE_PATH")?.takeIf { it.isNotBlank() }
    signingConfigs {
        if (releaseKeystore != null) {
            create("release") {
                storeFile = file(releaseKeystore)
                storeType = "pkcs12"
                storePassword = System.getenv("VOX_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("VOX_KEY_ALIAS") ?: "vox"
                keyPassword = System.getenv("VOX_KEYSTORE_PASSWORD")
            }
        }
    }

    // The AI-chip runtime (tool/fetch_qnn.sh) loads its DSP library from real
    // files: native libraries must be extracted at install, not read from
    // inside the APK.
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName(if (releaseKeystore != null) "release" else "debug")
            // Several plugins rely on reflection; skip shrinking for reliability.
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.4")
    // Media notification + lock-screen / headset controls.
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.media:media:1.7.0")
}

flutter {
    source = "../.."
}
