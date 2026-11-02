// G1 microphone audio: LC3, 16 kHz mono, 10 ms frames of 20 bytes
// (same parameters as the Even Realities demo). Unlike the demo, one decoder
// lives for the whole utterance, so packet loss concealment keeps working
// across packets.
#include <jni.h>
#include <cstdlib>
#include <cstdint>
#include "include/lc3.h"

static const int kDtUs = 10000;
static const int kSrHz = 16000;
static const int kFrameBytes = 20;

struct Decoder {
    lc3_decoder_t dec;
    void *mem;
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_vox_evensuite_lc3_Lc3_open(JNIEnv *, jclass) {
    auto *d = static_cast<Decoder *>(calloc(1, sizeof(Decoder)));
    if (!d) return 0;
    d->mem = malloc(lc3_decoder_size(kDtUs, kSrHz));
    if (!d->mem) {
        free(d);
        return 0;
    }
    d->dec = lc3_setup_decoder(kDtUs, kSrHz, 0, d->mem);
    return reinterpret_cast<jlong>(d);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_vox_evensuite_lc3_Lc3_decode(JNIEnv *env, jclass, jlong handle, jbyteArray input) {
    auto *d = reinterpret_cast<Decoder *>(handle);
    const jsize n = env->GetArrayLength(input);
    const int frames = n / kFrameBytes;
    const int samples = lc3_frame_samples(kDtUs, kSrHz);
    jbyteArray out = env->NewByteArray(frames * samples * 2);
    if (!d || frames == 0 || samples <= 0) return out;
    jbyte *src = env->GetByteArrayElements(input, nullptr);
    auto *pcm = static_cast<int16_t *>(malloc(samples * sizeof(int16_t)));
    for (int f = 0; f < frames; f++) {
        lc3_decode(d->dec, src + f * kFrameBytes, kFrameBytes, LC3_PCM_FORMAT_S16, pcm, 1);
        env->SetByteArrayRegion(out, f * samples * 2, samples * 2, reinterpret_cast<jbyte *>(pcm));
    }
    free(pcm);
    env->ReleaseByteArrayElements(input, src, JNI_ABORT);
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_vox_evensuite_lc3_Lc3_close(JNIEnv *, jclass, jlong handle) {
    auto *d = reinterpret_cast<Decoder *>(handle);
    if (!d) return;
    free(d->mem);
    free(d);
}
