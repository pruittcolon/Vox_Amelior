/// A minimal ONNX Runtime binding (float tensors in, float tensors out).
///
/// The speech engine already ships ONNX Runtime as its own library
/// (libonnxruntime.so), so models it does not know about, like the Clarity
/// upscaler, run on the same optimized runtime through its C API. The C API
/// is a table of function pointers whose order never changes between
/// versions; the indices below come from onnxruntime_c_api.h.
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as so;

/// The oldest API version with everything used here; any newer runtime serves it.
const _apiVersion = 17;

typedef _P = Pointer<Void>;
typedef _PP = Pointer<Pointer<Void>>;

final class _ApiBase extends Struct {
  external Pointer<NativeFunction<_PP Function(Uint32)>> getApi;
  external Pointer<NativeFunction<Pointer<Utf8> Function()>> getVersionString;
}

class OrtError implements Exception {
  OrtError(this.message);
  final String message;
  @override
  String toString() => 'ONNX Runtime: $message';
}

/// One loaded runtime per isolate.
class Ort {
  Ort._(this._api, this.version) {
    final out = calloc<_P>();
    final id = 'narrator'.toNativeUtf8();
    try {
      check(_createEnv(3, id, out)); // log level 3: errors only
      _env = out.value;
      check(_createCpuMemoryInfo(1, 0, out)); // arena allocator, default memory
      _memory = out.value;
      check(_getAllocator(out));
      _allocator = out.value;
    } finally {
      calloc.free(out);
      calloc.free(id);
    }
  }

  static Ort? _instance;

  /// Loads the runtime that ships with the speech engine.
  static Ort get instance => _instance ??= _load();

  static Ort _load() {
    DynamicLibrary lib;
    if (Platform.isAndroid) {
      lib = DynamicLibrary.open('libonnxruntime.so');
    } else if (Platform.isLinux) {
      // Command-line tools: next to the engine's library in the pub cache.
      final dir = so.resolveSherpaOnnxPathSync();
      lib = DynamicLibrary.open(dir == null ? 'libonnxruntime.so' : '$dir/libonnxruntime.so');
    } else if (Platform.isMacOS || Platform.isIOS) {
      lib = DynamicLibrary.process();
    } else {
      lib = DynamicLibrary.open('onnxruntime.dll');
    }
    final base = lib.lookupFunction<Pointer<_ApiBase> Function(), Pointer<_ApiBase> Function()>('OrtGetApiBase')();
    final version = base.ref.getVersionString.asFunction<Pointer<Utf8> Function()>()().toDartString();
    final api = base.ref.getApi.asFunction<_PP Function(int)>()(_apiVersion);
    if (api == nullptr) throw OrtError('runtime $version does not offer API $_apiVersion');
    return Ort._(api, version);
  }

  final _PP _api;
  final String version;
  late final _P _env;
  late final _P _memory;
  late final _P _allocator;

  // The functions used, by their index in the OrtApi table.
  late final _errorMessage =
      _api[2].cast<NativeFunction<Pointer<Utf8> Function(_P)>>().asFunction<Pointer<Utf8> Function(_P)>();
  late final _createEnv = _api[3]
      .cast<NativeFunction<_P Function(Int32, Pointer<Utf8>, _PP)>>()
      .asFunction<_P Function(int, Pointer<Utf8>, _PP)>();
  late final _createSession = _api[7]
      .cast<NativeFunction<_P Function(_P, Pointer<Utf8>, _P, _PP)>>()
      .asFunction<_P Function(_P, Pointer<Utf8>, _P, _PP)>();
  late final _run = _api[9]
      .cast<NativeFunction<_P Function(_P, _P, Pointer<Pointer<Utf8>>, _PP, Size, Pointer<Pointer<Utf8>>, Size, _PP)>>()
      .asFunction<_P Function(_P, _P, Pointer<Pointer<Utf8>>, _PP, int, Pointer<Pointer<Utf8>>, int, _PP)>();
  late final _createSessionOptions = _api[10].cast<NativeFunction<_P Function(_PP)>>().asFunction<_P Function(_PP)>();
  late final _setGraphOptimizationLevel =
      _api[23].cast<NativeFunction<_P Function(_P, Int32)>>().asFunction<_P Function(_P, int)>();
  late final _setIntraOpNumThreads = _api[24].cast<NativeFunction<_P Function(_P, Int32)>>().asFunction<_P Function(_P, int)>();
  late final _setInterOpNumThreads = _api[25].cast<NativeFunction<_P Function(_P, Int32)>>().asFunction<_P Function(_P, int)>();
  late final _sessionGetInputCount =
      _api[30].cast<NativeFunction<_P Function(_P, Pointer<Size>)>>().asFunction<_P Function(_P, Pointer<Size>)>();
  late final _sessionGetOutputCount =
      _api[31].cast<NativeFunction<_P Function(_P, Pointer<Size>)>>().asFunction<_P Function(_P, Pointer<Size>)>();
  late final _sessionGetInputName = _api[36]
      .cast<NativeFunction<_P Function(_P, Size, _P, Pointer<Pointer<Utf8>>)>>()
      .asFunction<_P Function(_P, int, _P, Pointer<Pointer<Utf8>>)>();
  late final _sessionGetOutputName = _api[37]
      .cast<NativeFunction<_P Function(_P, Size, _P, Pointer<Pointer<Utf8>>)>>()
      .asFunction<_P Function(_P, int, _P, Pointer<Pointer<Utf8>>)>();
  late final _createTensorWithData = _api[49]
      .cast<NativeFunction<_P Function(_P, _P, Size, Pointer<Int64>, Size, Int32, _PP)>>()
      .asFunction<_P Function(_P, _P, int, Pointer<Int64>, int, int, _PP)>();
  late final _getTensorMutableData = _api[51].cast<NativeFunction<_P Function(_P, _PP)>>().asFunction<_P Function(_P, _PP)>();
  late final _getDimensionsCount =
      _api[61].cast<NativeFunction<_P Function(_P, Pointer<Size>)>>().asFunction<_P Function(_P, Pointer<Size>)>();
  late final _getDimensions = _api[62]
      .cast<NativeFunction<_P Function(_P, Pointer<Int64>, Size)>>()
      .asFunction<_P Function(_P, Pointer<Int64>, int)>();
  late final _getTensorTypeAndShape = _api[65].cast<NativeFunction<_P Function(_P, _PP)>>().asFunction<_P Function(_P, _PP)>();
  late final _createCpuMemoryInfo =
      _api[69].cast<NativeFunction<_P Function(Int32, Int32, _PP)>>().asFunction<_P Function(int, int, _PP)>();
  late final _allocatorFree = _api[76].cast<NativeFunction<_P Function(_P, _P)>>().asFunction<_P Function(_P, _P)>();
  late final _getAllocator = _api[78].cast<NativeFunction<_P Function(_PP)>>().asFunction<_P Function(_PP)>();
  late final _releaseStatus = _api[93].cast<NativeFunction<Void Function(_P)>>().asFunction<void Function(_P)>();
  late final _releaseSession = _api[95].cast<NativeFunction<Void Function(_P)>>().asFunction<void Function(_P)>();
  late final _releaseValue = _api[96].cast<NativeFunction<Void Function(_P)>>().asFunction<void Function(_P)>();
  late final _releaseTypeAndShape = _api[99].cast<NativeFunction<Void Function(_P)>>().asFunction<void Function(_P)>();
  late final _releaseSessionOptions = _api[100].cast<NativeFunction<Void Function(_P)>>().asFunction<void Function(_P)>();

  void check(_P status) {
    if (status == nullptr) return;
    final msg = _errorMessage(status).toDartString();
    _releaseStatus(status);
    throw OrtError(msg);
  }

  /// Opens a model file (weights in a `.data` file next to it are found too).
  OrtSession open(String path, {int threads = 2}) {
    final out = calloc<_P>();
    final cPath = path.toNativeUtf8();
    try {
      check(_createSessionOptions(out));
      final opts = out.value;
      try {
        check(_setIntraOpNumThreads(opts, threads));
        check(_setInterOpNumThreads(opts, 1));
        check(_setGraphOptimizationLevel(opts, 99)); // every optimization
        check(_createSession(_env, cPath, opts, out));
      } finally {
        _releaseSessionOptions(opts);
      }
      final s = out.value;
      return OrtSession._(this, s, _names(s, _sessionGetInputCount, _sessionGetInputName),
          _names(s, _sessionGetOutputCount, _sessionGetOutputName));
    } finally {
      calloc.free(out);
      calloc.free(cPath);
    }
  }

  List<String> _names(
    _P session,
    _P Function(_P, Pointer<Size>) count,
    _P Function(_P, int, _P, Pointer<Pointer<Utf8>>) get,
  ) {
    final n = calloc<Size>();
    final name = calloc<Pointer<Utf8>>();
    try {
      check(count(session, n));
      final names = <String>[];
      for (var i = 0; i < n.value; i++) {
        check(get(session, i, _allocator, name));
        names.add(name.value.toDartString());
        check(_allocatorFree(_allocator, name.value.cast()));
      }
      return names;
    } finally {
      calloc.free(n);
      calloc.free(name);
    }
  }
}

/// A float tensor in native memory (an ONNX Runtime value).
class OrtTensor {
  OrtTensor._(this._ort, this._value, this.shape, this._owned);

  final Ort _ort;
  final _P _value;
  final List<int> shape;
  final Pointer<Float>? _owned;
  bool _freed = false;

  int get length => shape.fold(1, (a, b) => a * b);

  /// A view of the data; valid until [free].
  Float32List get data {
    final p = calloc<_P>();
    try {
      _ort.check(_ort._getTensorMutableData(_value, p));
      return p.value.cast<Float>().asTypedList(length);
    } finally {
      calloc.free(p);
    }
  }

  void free() {
    if (_freed) return;
    _freed = true;
    _ort._releaseValue(_value);
    if (_owned != null) calloc.free(_owned);
  }
}

class OrtSession {
  OrtSession._(this._ort, this._session, this.inputs, this.outputs);

  final Ort _ort;
  final _P _session;
  final List<String> inputs;
  final List<String> outputs;

  /// Copies [data] into a native tensor of [shape].
  OrtTensor tensor(Float32List data, List<int> shape) {
    final buf = calloc<Float>(data.length);
    buf.asTypedList(data.length).setAll(0, data);
    final dims = calloc<Int64>(shape.length);
    final out = calloc<_P>();
    try {
      for (var i = 0; i < shape.length; i++) {
        dims[i] = shape[i];
      }
      // Element type 1 = float32.
      _ort.check(_ort._createTensorWithData(_ort._memory, buf.cast(), data.length * 4, dims, shape.length, 1, out));
      return OrtTensor._(_ort, out.value, List.unmodifiable(shape), buf);
    } catch (_) {
      calloc.free(buf);
      rethrow;
    } finally {
      calloc.free(dims);
      calloc.free(out);
    }
  }

  /// Runs the model on [feeds] (one per input, in order) and returns every output.
  List<OrtTensor> run(List<OrtTensor> feeds) {
    final inNames = calloc<Pointer<Utf8>>(inputs.length);
    final outNames = calloc<Pointer<Utf8>>(outputs.length);
    final inVals = calloc<_P>(feeds.length);
    final outVals = calloc<_P>(outputs.length);
    try {
      for (var i = 0; i < inputs.length; i++) {
        inNames[i] = inputs[i].toNativeUtf8();
      }
      for (var i = 0; i < outputs.length; i++) {
        outNames[i] = outputs[i].toNativeUtf8();
      }
      for (var i = 0; i < feeds.length; i++) {
        inVals[i] = feeds[i]._value;
      }
      _ort.check(_ort._run(_session, nullptr, inNames, inVals, feeds.length, outNames, outputs.length, outVals));
      return [for (var i = 0; i < outputs.length; i++) OrtTensor._(_ort, outVals[i], _shape(outVals[i]), null)];
    } finally {
      for (var i = 0; i < inputs.length; i++) {
        if (inNames[i] != nullptr) calloc.free(inNames[i]);
      }
      for (var i = 0; i < outputs.length; i++) {
        if (outNames[i] != nullptr) calloc.free(outNames[i]);
      }
      calloc.free(inNames);
      calloc.free(outNames);
      calloc.free(inVals);
      calloc.free(outVals);
    }
  }

  List<int> _shape(_P value) {
    final info = calloc<_P>();
    final n = calloc<Size>();
    try {
      _ort.check(_ort._getTensorTypeAndShape(value, info));
      try {
        _ort.check(_ort._getDimensionsCount(info.value, n));
        final dims = calloc<Int64>(n.value);
        try {
          _ort.check(_ort._getDimensions(info.value, dims, n.value));
          return List.unmodifiable([for (var i = 0; i < n.value; i++) dims[i]]);
        } finally {
          calloc.free(dims);
        }
      } finally {
        _ort._releaseTypeAndShape(info.value);
      }
    } finally {
      calloc.free(info);
      calloc.free(n);
    }
  }

  void free() => _ort._releaseSession(_session);
}
