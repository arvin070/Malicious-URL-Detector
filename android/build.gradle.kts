plugins {
    // AGP 9 has built-in Kotlin support. Applying org.jetbrains.kotlin.android
    // on top of it fails with "extension with name 'kotlin' already registered".
    id("com.android.application") version "9.3.2" apply false
}
