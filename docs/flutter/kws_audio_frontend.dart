import 'dart:math' as math;
import 'dart:typed_data';

// import 'package:fftea/fftea.dart';

/// Converts PCM16 mono audio to MFCC features that match main_preset_model.py.
///
/// Training-side settings mirrored here:
/// - sample rate: 16_000
/// - duration: 1 second (16_000 samples)
/// - n_fft: 512
/// - hop_length: 160
/// - n_mels: 40
/// - n_mfcc: 13
///
/// Output shape is [1, 101, 13] (batch, frames, mfcc).
class KwsAudioFrontend {
  static const int sampleRate = 16000;
  static const int durationSeconds = 1;
  static const int targetSamples = sampleRate * durationSeconds;

  static const int nFft = 512;
  static const int hopLength = 160;
  static const int nMels = 40;
  static const int nMfcc = 13;

  static const double _eps = 1e-10;
  static const double _normEps = 1e-6;

  // late final FFT _fft;
  late final List<double> _hannWindow;
  late final List<List<double>> _melFilterBank;

  KwsAudioFrontend() {
    // _fft = FFT(nFft);
    _hannWindow = _buildHannWindow(nFft);
    _melFilterBank = _buildMelFilterBank();
  }

  /// Returns a tensor shaped [1, frames, 13] suitable for tflite_flutter.
  List<List<List<double>>> pcm16BytesToInputTensor(Uint8List pcm16leBytes) {
    final waveform = _pcm16LeToWaveform(pcm16leBytes);
    final fixed = _fixLength(waveform, targetSamples);
    final mfcc = _extractMfcc(fixed);

    final frames = mfcc.length;
    final tensor = List.generate(
      1,
      (_) => List.generate(
        frames,
        (t) => List<double>.from(mfcc[t]),
        growable: false,
      ),
      growable: false,
    );

    return tensor;
  }

  /// Same content as [pcm16BytesToInputTensor], flattened for manual buffer use.
  Float32List pcm16BytesToFlattened(Uint8List pcm16leBytes) {
    final tensor = pcm16BytesToInputTensor(pcm16leBytes)[0];
    final flat = Float32List(tensor.length * nMfcc);
    var idx = 0;
    for (var t = 0; t < tensor.length; t++) {
      for (var c = 0; c < nMfcc; c++) {
        flat[idx++] = tensor[t][c].toDouble();
      }
    }
    return flat;
  }

  List<double> _pcm16LeToWaveform(Uint8List bytes) {
    final byteData = ByteData.sublistView(bytes);
    final sampleCount = bytes.length ~/ 2;
    final out = List<double>.filled(sampleCount, 0.0, growable: false);

    for (var i = 0; i < sampleCount; i++) {
      final v = byteData.getInt16(i * 2, Endian.little);
      out[i] = v / 32768.0;
    }

    return out;
  }

  List<double> _fixLength(List<double> x, int targetLen) {
    if (x.length == targetLen) return x;

    final out = List<double>.filled(targetLen, 0.0, growable: false);
    if (x.length > targetLen) {
      for (var i = 0; i < targetLen; i++) {
        out[i] = x[i];
      }
      return out;
    }

    for (var i = 0; i < x.length; i++) {
      out[i] = x[i];
    }
    return out;
  }

  List<List<double>> _extractMfcc(List<double> signal) {
    final padded = _reflectPad(signal, nFft ~/ 2);
    final frameCount = 1 + ((padded.length - nFft) ~/ hopLength);

    final mfccFrames = List.generate(
      frameCount,
      (_) => List<double>.filled(nMfcc, 0.0, growable: false),
      growable: false,
    );

    for (var frame = 0; frame < frameCount; frame++) {
      final start = frame * hopLength;
      final windowed = List<double>.filled(nFft, 0.0, growable: false);

      for (var i = 0; i < nFft; i++) {
        windowed[i] = padded[start + i] * _hannWindow[i];
      }

      // final spectrum = _fft.realFft(windowed);
      // final mags = spectrum.discardConjugates().magnitudes().toList(
      //   growable: false,
      // );

      final power = List<double>.filled(nFft ~/ 2 + 1, 0.0, growable: false);
      for (var k = 0; k < power.length; k++) {
        // final m = mags[k];
        // power[k] = m * m;
      }

      final melEnergies = List<double>.filled(nMels, 0.0, growable: false);
      for (var m = 0; m < nMels; m++) {
        double sum = 0.0;
        final filter = _melFilterBank[m];
        for (var k = 0; k < filter.length; k++) {
          sum += filter[k] * power[k];
        }
        melEnergies[m] = math.log(math.max(sum, _eps));
      }

      for (var c = 0; c < nMfcc; c++) {
        double sum = 0.0;
        for (var m = 0; m < nMels; m++) {
          sum += melEnergies[m] * math.cos(math.pi * c * (m + 0.5) / nMels);
        }
        mfccFrames[frame][c] = sum;
      }
    }

    // Match Python-side normalization: global mean/std over the MFCC matrix.
    double mean = 0.0;
    double count = 0.0;
    for (var t = 0; t < mfccFrames.length; t++) {
      for (var c = 0; c < nMfcc; c++) {
        mean += mfccFrames[t][c];
        count += 1.0;
      }
    }
    mean /= math.max(count, 1.0);

    double variance = 0.0;
    for (var t = 0; t < mfccFrames.length; t++) {
      for (var c = 0; c < nMfcc; c++) {
        final d = mfccFrames[t][c] - mean;
        variance += d * d;
      }
    }
    variance /= math.max(count, 1.0);
    final std = math.sqrt(variance);

    for (var t = 0; t < mfccFrames.length; t++) {
      for (var c = 0; c < nMfcc; c++) {
        mfccFrames[t][c] = (mfccFrames[t][c] - mean) / (std + _normEps);
      }
    }

    return mfccFrames;
  }

  List<double> _reflectPad(List<double> x, int pad) {
    final n = x.length;
    final out = List<double>.filled(n + pad * 2, 0.0, growable: false);

    for (var i = 0; i < n; i++) {
      out[pad + i] = x[i];
    }

    for (var i = 0; i < pad; i++) {
      final leftIdx = math.min(pad - i, n - 1);
      final rightIdx = math.max(n - 2 - i, 0);
      out[i] = x[leftIdx];
      out[pad + n + i] = x[rightIdx];
    }

    return out;
  }

  List<double> _buildHannWindow(int size) {
    final out = List<double>.filled(size, 0.0, growable: false);
    for (var i = 0; i < size; i++) {
      out[i] = 0.5 - 0.5 * math.cos((2.0 * math.pi * i) / (size - 1));
    }
    return out;
  }

  List<List<double>> _buildMelFilterBank() {
    final nBins = nFft ~/ 2 + 1;
    final filters = List.generate(
      nMels,
      (_) => List<double>.filled(nBins, 0.0, growable: false),
      growable: false,
    );

    final minMel = _hzToMel(0.0);
    final maxMel = _hzToMel(sampleRate / 2.0);

    final melPoints = List<double>.filled(nMels + 2, 0.0, growable: false);
    for (var i = 0; i < melPoints.length; i++) {
      melPoints[i] = minMel + (maxMel - minMel) * i / (nMels + 1);
    }

    final hzPoints = melPoints.map(_melToHz).toList(growable: false);
    final bins = hzPoints
        .map((hz) => ((nFft + 1) * hz / sampleRate).floor().clamp(0, nBins - 1))
        .toList(growable: false);

    for (var m = 1; m <= nMels; m++) {
      final left = bins[m - 1];
      final center = bins[m];
      final right = bins[m + 1];

      if (center > left) {
        for (var k = left; k < center; k++) {
          filters[m - 1][k] = (k - left) / (center - left);
        }
      }

      if (right > center) {
        for (var k = center; k < right; k++) {
          filters[m - 1][k] = (right - k) / (right - center);
        }
      }
    }

    return filters;
  }

  double _hzToMel(double hz) {
    return 2595.0 * _log10(1.0 + hz / 700.0);
  }

  double _melToHz(double mel) {
    final v = math.pow(10.0, mel / 2595.0).toDouble();
    return 700.0 * (v - 1.0);
  }

  double _log10(double x) {
    return math.log(x) / math.ln10;
  }
}
