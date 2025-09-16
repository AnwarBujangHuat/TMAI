import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class EnhancedHeyTMTester:
    def __init__(self, model_path='heytm_model.tflite', sample_rate=16000, duration=1.0, n_mfcc=13):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        
        # Load the TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors info
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ TFLite model loaded successfully from {model_path}")
            print(f"📋 Input shape: {self.input_details[0]['shape']}")
            print(f"📋 Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            self.interpreter = None
            return
        
        # Class mapping (should match your training)
        self.class_names = ['heytm', 'unknown', 'background']
        self.confidence_threshold = 0.5  # Adjustable threshold
        
        # Color scheme for better visualization
        self.colors = {
            'heytm': '#2E8B57',      # Sea Green
            'unknown': '#FF6B6B',    # Light Red
            'background': '#4ECDC4', # Teal
            'high_conf': '#27AE60',  # Green
            'low_conf': '#E74C3C'    # Red
        }
        
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file (same as training)"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or trim to fixed length
            if len(audio) > self.n_samples:
                audio = audio[:self.n_samples]
            else:
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=self.n_mfcc,
                n_fft=512,
                hop_length=160,
                n_mels=40
            )
            
            # Normalize
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
            
            return mfcc.T  # Transpose for time-first format
            
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            return None
    
    def predict_single(self, audio_path):
        """Predict keyword for a single audio file using TFLite model"""
        if self.interpreter is None:
            return None, 0.0, None
        
        features = self.extract_features(audio_path)
        if features is None:
            return None, 0.0, None
        
        # Prepare input data fo
        # r TFLite
        input_data = np.expand_dims(features, axis=0).astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        all_probabilities = predictions
        
        return self.class_names[predicted_class], float(confidence), all_probabilities.tolist()
    
    def test_folder(self, test_folder_path, output_file='test_results.json'):
        """Test all audio files in the test folder"""
        if not os.path.exists(test_folder_path):
            print(f"❌ Test folder not found: {test_folder_path}")
            return None
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_folder': test_folder_path,
            'model_info': {
                'confidence_threshold': self.confidence_threshold,
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'n_mfcc': self.n_mfcc
            },
            'file_results': [],
            'summary': {}
        }
        
        # Get all audio files
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        audio_files = []
        for file in os.listdir(test_folder_path):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(file)
        
        if not audio_files:
            print(f"❌ No audio files found in {test_folder_path}")
            return None
        
        print(f"🔍 Testing {len(audio_files)} audio files...")
        print("=" * 100)
        
        # Test each file
        for i, filename in enumerate(audio_files, 1):
            file_path = os.path.join(test_folder_path, filename)
            
            prediction, confidence, probabilities = self.predict_single(file_path)
            
            # Determine if it's likely a "heytm" file based on filename
            filename_lower = filename.lower()
            likely_heytm = any(keyword in filename_lower for keyword in ['heytm', 'hey_tm', 'hey-tm'])
            
            # Create result entry
            file_result = {
                'filename': filename,
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[j]: float(probabilities[j]) for j in range(len(self.class_names))
                },
                'likely_heytm_from_filename': likely_heytm,
                'high_confidence': confidence > self.confidence_threshold,
                'file_size': os.path.getsize(file_path) / 1024  # Size in KB
            }
            
            results['file_results'].append(file_result)
            
            # Enhanced progress display
            status_icon = "🟢" if prediction == 'heytm' and confidence > self.confidence_threshold else "🔴"
            conf_bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
            
            print(f"{status_icon} [{i:3d}/{len(audio_files)}] {filename[:35]:35} | "
                  f"{prediction:10} | {confidence:.3f} [{conf_bar}] | "
                  f"Expected: {'HeyTM' if likely_heytm else 'Other':5}")
        
        # Generate summary statistics
        results['summary'] = self._generate_summary(results['file_results'])
        
        # Save results to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 100)
        print(f"✅ Test completed! Results saved to {output_file}")
        
        return results
    
    def _generate_summary(self, file_results):
        """Generate comprehensive summary statistics from test results"""
        total_files = len(file_results)
        
        # Count predictions
        pred_counts = defaultdict(int)
        high_conf_counts = defaultdict(int)
        
        for result in file_results:
            pred_counts[result['prediction']] += 1
            if result['high_confidence']:
                high_conf_counts[result['prediction']] += 1
        
        # Files that likely contain "heytm" based on filename
        likely_heytm_files = [r for r in file_results if r['likely_heytm_from_filename']]
        likely_heytm_correct = [r for r in likely_heytm_files if r['prediction'] == 'heytm']
        
        # Files that don't contain "heytm" in filename
        likely_other_files = [r for r in file_results if not r['likely_heytm_from_filename']]
        likely_other_correct = [r for r in likely_other_files if r['prediction'] != 'heytm']
        
        # Calculate confusion matrix data if we have filename hints
        true_positives = len([r for r in likely_heytm_files if r['prediction'] == 'heytm'])
        false_positives = len([r for r in likely_other_files if r['prediction'] == 'heytm'])
        false_negatives = len([r for r in likely_heytm_files if r['prediction'] != 'heytm'])
        true_negatives = len([r for r in likely_other_files if r['prediction'] != 'heytm'])
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        summary = {
            'total_files': total_files,
            'predictions': dict(pred_counts),
            'high_confidence_predictions': dict(high_conf_counts),
            'confidence_stats': {
                'mean_confidence': np.mean([r['confidence'] for r in file_results]),
                'std_confidence': np.std([r['confidence'] for r in file_results]),
                'min_confidence': min([r['confidence'] for r in file_results]),
                'max_confidence': max([r['confidence'] for r in file_results]),
                'median_confidence': np.median([r['confidence'] for r in file_results])
            },
            'performance_metrics': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': (true_positives + true_negatives) / total_files if total_files > 0 else 0
            }
        }
        
        # Add accuracy estimates if we can infer from filenames
        if likely_heytm_files:
            summary['filename_based_analysis'] = {
                'likely_heytm_files': len(likely_heytm_files),
                'correctly_identified_heytm': len(likely_heytm_correct),
                'heytm_accuracy': len(likely_heytm_correct) / len(likely_heytm_files) if likely_heytm_files else 0
            }
        
        if likely_other_files:
            if 'filename_based_analysis' not in summary:
                summary['filename_based_analysis'] = {}
            summary['filename_based_analysis']['likely_other_files'] = len(likely_other_files)
            summary['filename_based_analysis']['correctly_identified_other'] = len(likely_other_correct)
            summary['filename_based_analysis']['other_accuracy'] = len(likely_other_correct) / len(likely_other_files) if likely_other_files else 0
        
        return summary
    
    def create_enhanced_visualizations(self, results, save_plots=True):
        """Create comprehensive visualizations of test results"""
        if not results:
            print("❌ No results to visualize")
            return
        
        file_results = results['file_results']
        df = pd.DataFrame(file_results)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Main title
        fig.suptitle('🎤 HeyTM Model Performance Analysis', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Prediction Distribution (Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        pred_counts = df['prediction'].value_counts()
        colors = [self.colors.get(pred, '#95A5A6') for pred in pred_counts.index]
        wedges, texts, autotexts = ax1.pie(pred_counts.values, labels=pred_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('🎯 Prediction Distribution', fontweight='bold')
        
        # 2. Confidence Distribution (Histogram)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(df['confidence'], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(self.confidence_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold ({self.confidence_threshold})')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Number of Files')
        ax2.set_title('📊 Confidence Score Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence by Prediction Class (Box Plot)
        ax3 = fig.add_subplot(gs[0, 2])
        box_plot = ax3.boxplot([df[df['prediction'] == pred]['confidence'].values 
                               for pred in self.class_names if pred in df['prediction'].values],
                              labels=[pred for pred in self.class_names if pred in df['prediction'].values],
                              patch_artist=True)
        
        for patch, class_name in zip(box_plot['boxes'], 
                                   [pred for pred in self.class_names if pred in df['prediction'].values]):
            patch.set_facecolor(self.colors.get(class_name, '#95A5A6'))
            patch.set_alpha(0.7)
        
        ax3.set_title('📦 Confidence by Class', fontweight='bold')
        ax3.set_ylabel('Confidence Score')
        ax3.grid(True, alpha=0.3)
        
        # 4. High Confidence Predictions (Bar Chart)
        ax4 = fig.add_subplot(gs[0, 3])
        high_conf_df = df[df['high_confidence']]
        if not high_conf_df.empty:
            high_conf_counts = high_conf_df['prediction'].value_counts()
            bars = ax4.bar(high_conf_counts.index, high_conf_counts.values,
                          color=[self.colors.get(pred, '#95A5A6') for pred in high_conf_counts.index])
            ax4.set_title(f'🔥 High Confidence (>{self.confidence_threshold})', fontweight='bold')
            ax4.set_ylabel('Number of Files')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No High Confidence\nPredictions', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        # 5. Confusion Matrix (if we have filename hints)
        ax5 = fig.add_subplot(gs[1, 0])
        if 'performance_metrics' in results['summary']:
            metrics = results['summary']['performance_metrics']
            confusion_data = np.array([[metrics['true_positives'], metrics['false_negatives']],
                                     [metrics['false_positives'], metrics['true_negatives']]])
            
            im = ax5.imshow(confusion_data, interpolation='nearest', cmap='Blues')
            ax5.set_title('🎯 Confusion Matrix\n(Based on Filenames)', fontweight='bold')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    ax5.text(j, i, confusion_data[i, j], ha='center', va='center',
                           color='white' if confusion_data[i, j] > confusion_data.max()/2 else 'black',
                           fontsize=14, fontweight='bold')
            
            ax5.set_xticks([0, 1])
            ax5.set_yticks([0, 1])
            ax5.set_xticklabels(['Predicted HeyTM', 'Predicted Other'])
            ax5.set_yticklabels(['Actual HeyTM', 'Actual Other'])
        else:
            ax5.text(0.5, 0.5, 'No Ground Truth\nAvailable', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        
        # 6. Performance Metrics (Text + Bars)
        ax6 = fig.add_subplot(gs[1, 1])
        if 'performance_metrics' in results['summary']:
            metrics = results['summary']['performance_metrics']
            metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
            metric_values = [metrics['precision'], metrics['recall'], 
                           metrics['f1_score'], metrics['accuracy']]
            
            bars = ax6.barh(metric_names, metric_values, 
                           color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60'])
            ax6.set_xlim(0, 1)
            ax6.set_title('📈 Performance Metrics', fontweight='bold')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, metric_values)):
                ax6.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', va='center', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No Performance\nMetrics Available', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        
        # 7. Confidence vs File Size Scatter Plot
        ax7 = fig.add_subplot(gs[1, 2])
        scatter = ax7.scatter(df['file_size'], df['confidence'], 
                             c=[self.colors.get(pred, '#95A5A6') for pred in df['prediction']],
                             alpha=0.6, s=50)
        ax7.set_xlabel('File Size (KB)')
        ax7.set_ylabel('Confidence Score')
        ax7.set_title('💾 Confidence vs File Size', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Top Confident Predictions per Class
        ax8 = fig.add_subplot(gs[1, 3])
        top_confident = df.nlargest(10, 'confidence')[['filename', 'prediction', 'confidence']]
        y_pos = np.arange(len(top_confident))
        bars = ax8.barh(y_pos, top_confident['confidence'],
                       color=[self.colors.get(pred, '#95A5A6') for pred in top_confident['prediction']])
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels([f"{row['filename'][:15]}...({row['prediction']})" 
                            for _, row in top_confident.iterrows()], fontsize=8)
        ax8.set_title('🏆 Top 10 Confident Predictions', fontweight='bold')
        ax8.set_xlabel('Confidence Score')
        
        # 9. Problematic Files (Low Confidence)
        ax9 = fig.add_subplot(gs[2, 0:2])
        problematic = df[df['confidence'] < 0.4].nsmallest(15, 'confidence')
        if not problematic.empty:
            y_pos = np.arange(len(problematic))
            bars = ax9.barh(y_pos, problematic['confidence'],
                           color='#E74C3C', alpha=0.7)
            ax9.set_yticks(y_pos)
            ax9.set_yticklabels([f"{row['filename'][:20]}...({row['prediction']})" 
                                for _, row in problematic.iterrows()], fontsize=8)
            ax9.set_title('⚠️ Files Needing Attention (Low Confidence)', fontweight='bold')
            ax9.set_xlabel('Confidence Score')
            ax9.axvline(0.4, color='orange', linestyle='--', alpha=0.7)
        else:
            ax9.text(0.5, 0.5, '✅ No Problematic Files Found!', 
                    ha='center', va='center', transform=ax9.transAxes, 
                    fontsize=14, fontweight='bold', color='green')
        
        # 10. Probability Distributions for Each Class
        ax10 = fig.add_subplot(gs[2, 2:4])
        for i, class_name in enumerate(self.class_names):
            probs = [r['probabilities'][class_name] for r in file_results]
            ax10.hist(probs, bins=20, alpha=0.5, label=f'{class_name.capitalize()}',
                     color=self.colors.get(class_name, '#95A5A6'))
        
        ax10.set_xlabel('Probability Score')
        ax10.set_ylabel('Frequency')
        ax10.set_title('🎲 Probability Distributions by Class', fontweight='bold')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('enhanced_test_results.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print("📊 Enhanced visualization saved as 'enhanced_test_results.png'")
        
        plt.show()
    
    def print_enhanced_summary(self, results):
        """Print a beautifully formatted comprehensive summary"""
        if not results:
            print("❌ No results to summarize")
            return
        
        summary = results['summary']
        
        print("\n" + "🎤" + "=" * 78 + "🎤")
        print("🎯 ENHANCED HEYTM MODEL TEST RESULTS")
        print("🎤" + "=" * 78 + "🎤")
        
        # Basic Info Table
        basic_info = [
            ["📅 Test Date", results['timestamp'][:19]],
            ["📁 Test Folder", results['test_folder']],
            ["🎯 Total Files", summary['total_files']],
            ["🎛️ Confidence Threshold", f"{self.confidence_threshold:.2f}"],
            ["🎵 Sample Rate", f"{self.sample_rate} Hz"],
            ["⏱️ Duration", f"{self.duration} sec"],
            ["🎚️ MFCC Features", self.n_mfcc]
        ]
        
        print("\n📋 TEST CONFIGURATION")
        print(tabulate(basic_info, tablefmt="grid", colalign=("left", "left")))
        
        # Prediction Results Table
        print("\n🔮 PREDICTION RESULTS")
        pred_table = []
        total_files = summary['total_files']
        for class_name in self.class_names:
            count = summary['predictions'].get(class_name, 0)
            high_conf = summary['high_confidence_predictions'].get(class_name, 0)
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            pred_table.append([
                f"{class_name.capitalize()} 🎯",
                f"{count:3d}",
                f"{percentage:5.1f}%",
                f"{high_conf:3d}",
                f"{'█' * int(percentage/5)}"
            ])
        
        print(tabulate(pred_table, 
                      headers=["Class", "Count", "Percentage", "High Conf", "Visual"],
                      tablefmt="grid", colalign=("left", "center", "center", "center", "left")))
        
        # Confidence Statistics
        print("\n📊 CONFIDENCE STATISTICS")
        conf_stats = summary['confidence_stats']
        conf_table = [
            ["📈 Mean", f"{conf_stats['mean_confidence']:.3f}"],
            ["📊 Median", f"{conf_stats['median_confidence']:.3f}"],
            ["📉 Std Dev", f"{conf_stats['std_confidence']:.3f}"],
            ["⬇️ Minimum", f"{conf_stats['min_confidence']:.3f}"],
            ["⬆️ Maximum", f"{conf_stats['max_confidence']:.3f}"]
        ]
        print(tabulate(conf_table, tablefmt="grid", colalign=("left", "center")))
        
        # Performance Metrics
        if 'performance_metrics' in summary:
            print("\n🏆 PERFORMANCE METRICS")
            metrics = summary['performance_metrics']
            
            perf_table = [
                ["✅ True Positives", metrics['true_positives']],
                ["❌ False Positives", metrics['false_positives']],
                ["⚠️ False Negatives", metrics['false_negatives']],
                ["✅ True Negatives", metrics['true_negatives']],
                ["", ""],
                ["🎯 Precision", f"{metrics['precision']:.3f}"],
                ["🔍 Recall", f"{metrics['recall']:.3f}"],
                ["⚖️ F1-Score", f"{metrics['f1_score']:.3f}"],
                ["📊 Accuracy", f"{metrics['accuracy']:.3f}"]
            ]
            print(tabulate(perf_table, tablefmt="grid", colalign=("left", "center")))
        
        # Filename-based Analysis
        if 'filename_based_analysis' in summary:
            print("\n🏷️ FILENAME-BASED ANALYSIS")
            fb_analysis = summary['filename_based_analysis']
            
            fb_table = []
            if 'heytm_accuracy' in fb_analysis:
                fb_table.extend([
                    ["🎤 Expected HeyTM Files", fb_analysis['likely_heytm_files']],
                    ["✅ Correctly Identified", fb_analysis['correctly_identified_heytm']],
                    ["📊 HeyTM Accuracy", f"{fb_analysis['heytm_accuracy']*100:.1f}%"]
                ])
            
            if 'other_accuracy' in fb_analysis:
                fb_table.extend([
                    ["🚫 Expected Other Files", fb_analysis['likely_other_files']],
                    ["✅ Correctly Identified", fb_analysis['correctly_identified_other']],
                    ["📊 Other Accuracy", f"{fb_analysis['other_accuracy']*100:.1f}%"]
                ])
            
            print(tabulate(fb_table, tablefmt="grid", colalign=("left", "center")))
        
        print("\n" + "🎤" + "=" * 78 + "🎤")
    
    def generate_detailed_report(self, results, output_file='detailed_report.html'):
        """Generate a detailed HTML report"""
        if not results:
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HeyTM Model Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                .file-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }}
                .file-card {{ background: white; padding: 15px; border-radius: 8px; border-left: 4px solid; }}
                .heytm {{ border-left-color: #27ae60; }}
                .unknown {{ border-left-color: #e74c3c; }}
                .background {{ border-left-color: #3498db; }}
                .confidence-bar {{ width: 100%; height: 20px; background-color: #ecf0f1; border-radius: 10px; overflow: hidden; }}
                .confidence-fill {{ height: 100%; background-color: #27ae60; transition: width 0.3s; }}
                .low-confidence {{ background-color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎤 HeyTM Model Test Report</h1>
                <p>Generated on {results['timestamp']}</p>
            </div>
        """
        
        # Add summary section
        summary = results['summary']
        html_content += f"""
            <div class="summary">
                <h2>📊 Summary Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Files Tested</td><td>{summary['total_files']}</td></tr>
                    <tr><td>Mean Confidence</td><td>{summary['confidence_stats']['mean_confidence']:.3f}</td></tr>
                    <tr><td>HeyTM Predictions</td><td>{summary['predictions'].get('heytm', 0)}</td></tr>
                    <tr><td>High Confidence Predictions</td><td>{sum(summary['high_confidence_predictions'].values())}</td></tr>
                </table>
            </div>
            
            <div class="summary">
                <h2>📁 Individual File Results</h2>
                <div class="file-grid">
        """
        
        # Add individual file results
        for result in results['file_results']:
            conf_width = int(result['confidence'] * 100)
            conf_class = 'low-confidence' if result['confidence'] < self.confidence_threshold else ''
            
            html_content += f"""
                <div class="file-card {result['prediction']}">
                    <h4>{result['filename']}</h4>
                    <p><strong>Prediction:</strong> {result['prediction'].capitalize()}</p>
                    <p><strong>Confidence:</strong> {result['confidence']:.3f}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill {conf_class}" style="width: {conf_width}%"></div>
                    </div>
                    <p><strong>Expected HeyTM:</strong> {'Yes' if result['likely_heytm_from_filename'] else 'No'}</p>
                    <p><strong>File Size:</strong> {result['file_size']:.1f} KB</p>
                </div>
            """
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Detailed HTML report saved as '{output_file}'")
    
    def find_problematic_files(self, results, min_confidence=0.3):
        """Identify files that might need attention with enhanced analysis"""
        if not results:
            return
        
        print(f"\n🚨 FILES NEEDING ATTENTION (confidence < {min_confidence})")
        print("=" * 90)
        
        problematic = [r for r in results['file_results'] if r['confidence'] < min_confidence]
        
        if not problematic:
            print("🎉 Excellent! No problematic files found!")
            print("✅ All files have confidence scores above the threshold.")
            return
        
        # Sort by confidence (lowest first)
        problematic.sort(key=lambda x: x['confidence'])
        
        prob_table = []
        for i, result in enumerate(problematic, 1):
            expected = "HeyTM" if result['likely_heytm_from_filename'] else "Other"
            status = "❌ MISMATCH" if (expected == "HeyTM") != (result['prediction'] == 'heytm') else "✅ Match"
            
            prob_table.append([
                f"{i:2d}",
                result['filename'][:35] + "..." if len(result['filename']) > 35 else result['filename'],
                result['prediction'].capitalize(),
                f"{result['confidence']:.3f}",
                expected,
                status,
                "🔧" if result['confidence'] < 0.2 else "⚠️"
            ])
        
        print(tabulate(prob_table,
                      headers=["#", "Filename", "Prediction", "Confidence", "Expected", "Status", "Priority"],
                      tablefmt="grid"))
        
        print(f"\n📈 RECOMMENDATIONS:")
        print(f"🔧 Critical (< 0.2 confidence): {len([r for r in problematic if r['confidence'] < 0.2])} files")
        print(f"⚠️ Warning (0.2-{min_confidence} confidence): {len([r for r in problematic if 0.2 <= r['confidence'] < min_confidence])} files")
        print(f"💡 Consider retraining with more diverse data for these edge cases.")
    
    def export_results_csv(self, results, output_file='test_results.csv'):
        """Export results to CSV for further analysis"""
        if not results:
            return
        
        df = pd.DataFrame(results['file_results'])
        
        # Add additional computed columns
        df['prediction_correct'] = df.apply(
            lambda row: (row['likely_heytm_from_filename'] and row['prediction'] == 'heytm') or 
                       (not row['likely_heytm_from_filename'] and row['prediction'] != 'heytm'),
            axis=1
        )
        df['confidence_category'] = pd.cut(df['confidence'], 
                                         bins=[0, 0.3, 0.5, 0.7, 1.0], 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Add probability columns
        for class_name in self.class_names:
            df[f'prob_{class_name}'] = df['probabilities'].apply(lambda x: x[class_name])
        
        # Remove the nested probabilities column for CSV
        df_export = df.drop('probabilities', axis=1)
        
        df_export.to_csv(output_file, index=False)
        print(f"📊 Results exported to '{output_file}' for further analysis")

def main():
    """Enhanced main testing function with comprehensive analysis"""
    print("🎤" + "=" * 58 + "🎤")
    print("🎯 ENHANCED HEYTM KEYWORD SPOTTING MODEL TESTER")
    print("🎤" + "=" * 58 + "🎤")
    
    # Initialize enhanced tester
    tester = EnhancedHeyTMTester(
        model_path='heytm_model.tflite',
        sample_rate=16000,
        duration=1.0,
        n_mfcc=13
    )
    
    if tester.interpreter is None:
        print("❌ Cannot proceed without a valid TFLite model")
        return
    
    # Test the folder
    test_folder = 'test'  # Update path if needed
    print(f"\n🔍 Scanning folder: {test_folder}")
    
    results = tester.test_folder(test_folder, output_file='enhanced_test_results.json')
    
    if results:
        print(f"\n{'='*80}")
        print("📊 GENERATING COMPREHENSIVE ANALYSIS...")
        print(f"{'='*80}")
        
        # Print enhanced summary
        tester.print_enhanced_summary(results)
        
        # Find and analyze problematic files
        tester.find_problematic_files(results, min_confidence=0.4)
        
        # Create enhanced visualizations
        tester.create_enhanced_visualizations(results, save_plots=True)
        
        # Generate detailed HTML report
        tester.generate_detailed_report(results, 'heytm_detailed_report.html')
        
        # Export to CSV for further analysis
        tester.export_results_csv(results, 'heytm_analysis.csv')
        
        # Final summary
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"📋 Files generated:")
        print(f"   • enhanced_test_results.json - Complete results data")
        print(f"   • enhanced_test_results.png  - Comprehensive visualizations")
        print(f"   • heytm_detailed_report.html - Interactive HTML report")
        print(f"   • heytm_analysis.csv         - Data for spreadsheet analysis")
        print(f"{'='*80}")
        
        # Quick performance summary
        if 'performance_metrics' in results['summary']:
            metrics = results['summary']['performance_metrics']
            print(f"🎯 QUICK PERFORMANCE SUMMARY:")
            print(f"   Accuracy: {metrics['accuracy']*100:.1f}% | "
                  f"Precision: {metrics['precision']*100:.1f}% | "
                  f"Recall: {metrics['recall']*100:.1f}% | "
                  f"F1-Score: {metrics['f1_score']*100:.1f}%")
    
    return results

if __name__ == "__main__":
    results = main()