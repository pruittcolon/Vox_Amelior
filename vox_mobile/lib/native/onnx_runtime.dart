import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

typedef _Status = Pointer<Void>;
typedef _Release = void Function(Pointer<Void>);
typedef _ReleaseNative = Void Function(Pointer<Void>);

/// Minimal bindings to the ONNX Runtime C API, using the copy that ships
/// inside sherpa-onnx (1.28.2 on Android and Linux), so no second runtime is
/// bundled. Only what running one float model on the CPU needs.
///
/// `OrtApi` is a table of function pointers whose order never changes (new
/// functions are only appended), so functions are looked up by position.
/// Positions were read from onnxruntime_c_api.h (v1.28.2).
class OnnxRuntime {
  OnnxRuntime._(this._api);

  final Pointer<Pointer<Void>> _api;
  static OnnxRuntime? _instance;

  /// Any version up to the library's own works; 17 has everything used here.
  static const int _apiVersion = 17;

  // Positions in struct OrtApi.
  static const int _getErrorMessage = 2;
  static const int _createEnv = 3;
  static const int _createSession = 7;
  static const int _run = 9;
  static const int _createSessionOptions = 10;
  static const int _setGraphOptimizationLevel = 23;
  static const int _setIntraOpNumThreads = 24;
  static const int _createTensorWithDataAsOrtValue = 49;
  static const int _getTensorMutableData = 51;
  static const int _getDimensionsCount = 61;
  static const int _getDimensions = 62;
  static const int _getTensorTypeAndShape = 65;
  static const int _createCpuMemoryInfo = 69;
  static const int _releaseEnv = 92;
  static const int _releaseStatus = 93;
  static const int _releaseMemoryInfo = 94;
  static const int _releaseSession = 95;
  static const int _releaseValue = 96;
  static const int _releaseTensorTypeAndShapeInfo = 99;
  static const int _releaseSessionOptions = 100;

  /// Loads the runtime. [libraryDir] is only needed off Android (tests).
  static OnnxRuntime load({String? libraryDir}) {
    final existing = _instance;
    if (existing != null) return existing;
    final dir = libraryDir ?? Platform.environment['SHERPA_LIB_DIR'];
    final lib = DynamicLibrary.open(Platform.isAndroid || dir == null ? 'libonnxruntime.so' : '$dir/libonnxruntime.so');
    final getApiBase =
        lib.lookupFunction<Pointer<Pointer<Void>> Function(), Pointer<Pointer<Void>> Function()>('OrtGetApiBase');
    final getApi = getApiBase()[0]
        .cast<NativeFunction<Pointer<Pointer<Void>> Function(Uint32)>>()
        .asFunction<Pointer<Pointer<Void>> Function(int)>();
    final api = getApi(_apiVersion);
    if (api == nullptr) throw StateError('ONNX Runtime does not support API version $_apiVersion');
    return _instance = OnnxRuntime._(api);
  }

  Pointer<NativeFunction<N>> _at<N extends Function>(int index) => _api[index].cast<NativeFunction<N>>();

  late final _errorMessage =
      _at<Pointer<Utf8> Function(_Status)>(_getErrorMessage).asFunction<Pointer<Utf8> Function(_Status)>();
  late final _freeStatus = _at<_ReleaseNative>(_releaseStatus).asFunction<_Release>();
  late final _freeValue = _at<_ReleaseNative>(_releaseValue).asFunction<_Release>();
  late final _freeSession = _at<_ReleaseNative>(_releaseSession).asFunction<_Release>();
  late final _freeSessionOptions = _at<_ReleaseNative>(_releaseSessionOptions).asFunction<_Release>();
  late final _freeShapeInfo = _at<_ReleaseNative>(_releaseTensorTypeAndShapeInfo).asFunction<_Release>();
  late final _freeMemoryInfo = _at<_ReleaseNative>(_releaseMemoryInfo).asFunction<_Release>();
  late final _freeEnv = _at<_ReleaseNative>(_releaseEnv).asFunction<_Release>();

  /// Throws if [status] reports an error (and frees it).
  void check(Pointer<Void> status, String what) {
    if (status == nullptr) return;
    final message = _errorMessage(status).toDartString();
    _freeStatus(status);
    throw OnnxError('$what: $message');
  }

  late final Pointer<Void> env = () {
    final out = calloc<Pointer<Void>>();
    final name = 'vox'.toNativeUtf8();
    try {
      final create = _at<_Status Function(Int32, Pointer<Utf8>, Pointer<Pointer<Void>>)>(_createEnv)
          .asFunction<_Status Function(int, Pointer<Utf8>, Pointer<Pointer<Void>>)>();
      check(create(3 /* ORT_LOGGING_LEVEL_ERROR */, name, out), 'CreateEnv');
      return out.value;
    } finally {
      calloc
        ..free(out)
        ..free(name);
    }
  }();

  late final Pointer<Void> cpuMemory = () {
    final out = calloc<Pointer<Void>>();
    try {
      final create = _at<_Status Function(Int32, Int32, Pointer<Pointer<Void>>)>(_createCpuMemoryInfo)
          .asFunction<_Status Function(int, int, Pointer<Pointer<Void>>)>();
      check(create(0 /* OrtDeviceAllocator */, 0 /* OrtMemTypeDefault */, out), 'CreateCpuMemoryInfo');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }();

  Pointer<Void> createSession(String modelPath, {int threads = 2}) {
    final options = calloc<Pointer<Void>>();
    final session = calloc<Pointer<Void>>();
    final path = modelPath.toNativeUtf8();
    try {
      final createOptions = _at<_Status Function(Pointer<Pointer<Void>>)>(_createSessionOptions)
          .asFunction<_Status Function(Pointer<Pointer<Void>>)>();
      final setThreads = _at<_Status Function(Pointer<Void>, Int32)>(_setIntraOpNumThreads)
          .asFunction<_Status Function(Pointer<Void>, int)>();
      final setLevel = _at<_Status Function(Pointer<Void>, Int32)>(_setGraphOptimizationLevel)
          .asFunction<_Status Function(Pointer<Void>, int)>();
      final create = _at<_Status Function(Pointer<Void>, Pointer<Utf8>, Pointer<Void>, Pointer<Pointer<Void>>)>(_createSession)
          .asFunction<_Status Function(Pointer<Void>, Pointer<Utf8>, Pointer<Void>, Pointer<Pointer<Void>>)>();
      check(createOptions(options), 'CreateSessionOptions');
      check(setThreads(options.value, threads), 'SetIntraOpNumThreads');
      check(setLevel(options.value, 99 /* ORT_ENABLE_ALL */), 'SetSessionGraphOptimizationLevel');
      check(create(env, path, options.value, session), 'CreateSession');
      return session.value;
    } finally {
      if (options.value != nullptr) _freeSessionOptions(options.value);
      calloc
        ..free(options)
        ..free(session)
        ..free(path);
    }
  }

  void releaseSession(Pointer<Void> session) => _freeSession(session);

  late final _createTensor = _at<
          _Status Function(Pointer<Void>, Pointer<Void>, Size, Pointer<Int64>, Size, Int32, Pointer<Pointer<Void>>)>(
      _createTensorWithDataAsOrtValue)
      .asFunction<_Status Function(Pointer<Void>, Pointer<Void>, int, Pointer<Int64>, int, int, Pointer<Pointer<Void>>)>();

  late final _runFn = _at<
          _Status Function(Pointer<Void>, Pointer<Void>, Pointer<Pointer<Utf8>>, Pointer<Pointer<Void>>, Size,
              Pointer<Pointer<Utf8>>, Size, Pointer<Pointer<Void>>)>(_run)
      .asFunction<
          _Status Function(Pointer<Void>, Pointer<Void>, Pointer<Pointer<Utf8>>, Pointer<Pointer<Void>>, int,
              Pointer<Pointer<Utf8>>, int, Pointer<Pointer<Void>>)>();

  late final _typeAndShape = _at<_Status Function(Pointer<Void>, Pointer<Pointer<Void>>)>(_getTensorTypeAndShape)
      .asFunction<_Status Function(Pointer<Void>, Pointer<Pointer<Void>>)>();
  late final _dimsCount = _at<_Status Function(Pointer<Void>, Pointer<Size>)>(_getDimensionsCount)
      .asFunction<_Status Function(Pointer<Void>, Pointer<Size>)>();
  late final _dims = _at<_Status Function(Pointer<Void>, Pointer<Int64>, Size)>(_getDimensions)
      .asFunction<_Status Function(Pointer<Void>, Pointer<Int64>, int)>();
  late final _mutableData = _at<_Status Function(Pointer<Void>, Pointer<Pointer<Void>>)>(_getTensorMutableData)
      .asFunction<_Status Function(Pointer<Void>, Pointer<Pointer<Void>>)>();

  /// Runs [session] with float32 / int64 inputs; returns output [output] as
  /// a flat float list plus its shape.
  ({Float32List data, List<int> shape}) run(
    Pointer<Void> session, {
    required Map<String, ({Object data, List<int> shape})> inputs,
    required String output,
  }) {
    final arena = Arena();
    final values = <Pointer<Void>>[];
    var result = nullptr.cast<Void>();
    try {
      final names = arena<Pointer<Utf8>>(inputs.length);
      final tensors = arena<Pointer<Void>>(inputs.length);
      var i = 0;
      for (final e in inputs.entries) {
        names[i] = e.key.toNativeUtf8(allocator: arena);
        final rank = e.value.shape.length;
        final shape = arena<Int64>(rank == 0 ? 1 : rank);
        for (var d = 0; d < rank; d++) {
          shape[d] = e.value.shape[d];
        }
        final Pointer<Void> buffer;
        final int bytes;
        final int type;
        final data = e.value.data;
        if (data is Float32List) {
          final p = arena<Float>(data.isEmpty ? 1 : data.length);
          p.asTypedList(data.length).setAll(0, data);
          buffer = p.cast();
          bytes = data.length * 4;
          type = 1; // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        } else if (data is Int64List) {
          final p = arena<Int64>(data.isEmpty ? 1 : data.length);
          p.asTypedList(data.length).setAll(0, data);
          buffer = p.cast();
          bytes = data.length * 8;
          type = 7; // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        } else {
          throw ArgumentError('Unsupported input type ${data.runtimeType}');
        }
        final out = arena<Pointer<Void>>();
        check(_createTensor(cpuMemory, buffer, bytes, shape, rank, type, out), 'CreateTensor ${e.key}');
        values.add(out.value);
        tensors[i] = out.value;
        i++;
      }
      final outName = arena<Pointer<Utf8>>()..value = output.toNativeUtf8(allocator: arena);
      final outValue = arena<Pointer<Void>>();
      check(_runFn(session, nullptr, names, tensors, inputs.length, outName, 1, outValue), 'Run');
      result = outValue.value;

      final info = arena<Pointer<Void>>();
      check(_typeAndShape(result, info), 'GetTensorTypeAndShape');
      final List<int> shape;
      try {
        final count = arena<Size>();
        check(_dimsCount(info.value, count), 'GetDimensionsCount');
        final dims = arena<Int64>(count.value == 0 ? 1 : count.value);
        check(_dims(info.value, dims, count.value), 'GetDimensions');
        shape = [for (var d = 0; d < count.value; d++) dims[d]];
      } finally {
        _freeShapeInfo(info.value);
      }
      final elements = shape.fold<int>(1, (a, b) => a * b);
      final ptr = arena<Pointer<Void>>();
      check(_mutableData(result, ptr), 'GetTensorMutableData');
      return (data: Float32List.fromList(ptr.value.cast<Float>().asTypedList(elements)), shape: shape);
    } finally {
      for (final v in values) {
        _freeValue(v);
      }
      if (result != nullptr) _freeValue(result);
      arena.releaseAll();
    }
  }

  /// Frees the shared environment (normally never: it lives with the process).
  void dispose() {
    _freeMemoryInfo(cpuMemory);
    _freeEnv(env);
    _instance = null;
  }
}

class OnnxError implements Exception {
  const OnnxError(this.message);
  final String message;
  @override
  String toString() => 'OnnxError: $message';
}
