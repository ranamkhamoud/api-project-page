import gradio as gr
import os
from pipeline import AuthenticityDetectionPipeline
import traceback

# initialize the pipeline on startup
try:
    pipeline = AuthenticityDetectionPipeline(whisper_model_size="base")
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    pipeline_error = str(e)
    import traceback
    print(f"pipeline failed to start: {e}")
    traceback.print_exc()


# build the acoustic features display HTML
def build_acoustic_features_display(audio_class):
    classification = audio_class['classification']
    confidence = audio_class['confidence']
    cnn_class = audio_class['cnn_classification']
    cnn_conf = audio_class['cnn_confidence']
    prosody_class = audio_class['prosody_classification']
    prosody_conf = audio_class['prosody_confidence']
    prosody_scores = audio_class.get('prosody_scores', {})
    acoustic_features = audio_class.get('acoustic_features', {})
    
    if classification == 'spontaneous':
        main_color = '#10b981'
        bg_color = '#ecfdf5'
        label = 'SPONTANEOUS'
    else:
        main_color = '#f59e0b'
        bg_color = '#fffbeb'
        label = 'READ'
    
    cnn_color = '#10b981' if cnn_class == 'spontaneous' else '#f59e0b'
    prosody_color = '#10b981' if prosody_class == 'spontaneous' else '#f59e0b'
    
    # build main classification header
    output = f"""
<div style="background: linear-gradient(135deg, {bg_color} 0%, white 100%); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid {main_color}33;">
    <h3 style="margin: 0; color: {main_color}; font-size: 22px; font-weight: 700;">{label} SPEECH</h3>
    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 14px;">Combined acoustic analysis confidence: <strong>{confidence*100:.1f}%</strong></p>
</div>

<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Analysis Components</h4>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
        <div style="background: #f9fafb; border-radius: 10px; padding: 16px;">
            <div style="font-size: 12px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">CNN Neural Network</div>
            <div style="font-size: 20px; font-weight: 700; color: {cnn_color}; margin-bottom: 8px;">{cnn_class.upper()}</div>
            <div style="background: #e5e7eb; border-radius: 6px; overflow: hidden; height: 6px;">
                <div style="height: 100%; width: {cnn_conf*100:.0f}%; background: {cnn_color}; border-radius: 6px;"></div>
            </div>
            <div style="font-size: 11px; color: #9ca3af; margin-top: 6px;">{cnn_conf*100:.1f}% confidence</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 16px;">
            <div style="font-size: 12px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">Prosody Analysis</div>
            <div style="font-size: 20px; font-weight: 700; color: {prosody_color}; margin-bottom: 8px;">{prosody_class.upper()}</div>
            <div style="background: #e5e7eb; border-radius: 6px; overflow: hidden; height: 6px;">
                <div style="height: 100%; width: {prosody_conf*100:.0f}%; background: {prosody_color}; border-radius: 6px;"></div>
            </div>
            <div style="font-size: 11px; color: #9ca3af; margin-top: 6px;">{prosody_conf*100:.1f}% confidence</div>
        </div>
    </div>
</div>

<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Prosody Feature Breakdown</h4>
"""
    
    # feature descriptions
    feature_info = {
        'spectral_variability': {'name': 'Spectral Variability', 'unit': 'Hz', 'description': 'Variation in frequency content over time'},
        'zcr_mean': {'name': 'Zero Crossing Rate', 'unit': 'ratio', 'description': 'Rate of signal sign changes'},
        'energy_level': {'name': 'Energy Level', 'unit': 'RMS', 'description': 'Overall loudness and intensity'},
        'tempo': {'name': 'Speech Tempo', 'unit': 'BPM', 'description': 'Rhythmic pacing of speech'}
    }
    
    for key, info in feature_info.items():
        if key in prosody_scores:
            score_data = prosody_scores[key]
            score = score_data['score']
            value = score_data['value']
            interp = score_data['interpretation']
            unit = info['unit']
            
            bar_color = '#10b981' if score < 0.4 else '#f59e0b' if score > 0.6 else '#6b7280'
            indicator_position = score * 100
            
            output += f"""
    <div style="background: #f9fafb; border-radius: 10px; padding: 14px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div>
                <div style="font-weight: 600; color: #1f2937; font-size: 14px;">{info['name']}</div>
                <div style="font-size: 11px; color: #9ca3af;">{info['description']}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 13px; font-weight: 600; color: {bar_color};">{interp}</div>
                <div style="font-size: 11px; color: #6b7280;">{value:.3f} <span style="color: #9ca3af;">{unit}</span></div>
            </div>
        </div>
        <div style="position: relative; background: linear-gradient(to right, #10b981, #6b7280, #f59e0b); border-radius: 4px; height: 6px; margin: 10px 0 6px 0;">
            <div style="position: absolute; left: {indicator_position}%; top: -4px; transform: translateX(-50%); width: 14px; height: 14px; background: white; border: 2px solid {bar_color}; border-radius: 50%; box-shadow: 0 1px 3px rgba(0,0,0,0.15);"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #9ca3af;">
            <span>Spontaneous</span>
            <span>Read</span>
        </div>
    </div>
"""
    
    output += "</div>"
    
    output += """
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Raw Acoustic Measurements</h4>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
"""
    
    if acoustic_features:
        metrics = [
            ('Tempo', f"{acoustic_features.get('tempo', 0):.1f}", 'BPM'),
            ('Pitch Mean', f"{acoustic_features.get('pitch_mean', 0):.1f}", 'Hz'),
            ('Energy Mean', f"{acoustic_features.get('energy_mean', 0):.4f}", ''),
            ('ZCR Mean', f"{acoustic_features.get('zcr_mean', 0):.4f}", ''),
        ]
        for name, value, unit in metrics:
            output += f"""
        <div style="background: #f9fafb; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="font-size: 16px; font-weight: 600; color: #1f2937;">{value}</div>
            <div style="font-size: 10px; color: #6b7280; margin-top: 2px;">{name} {unit}</div>
        </div>
"""
    
    output += """
    </div>
</div>
"""
    
    return output


# build the transcription display HTML
def build_transcription_display(asr):
    if asr['speech_rate'] > 160:
        rate_color = '#f59e0b'
        rate_label = 'Fast'
        rate_desc = 'Above average speaking speed'
    elif asr['speech_rate'] < 120:
        rate_color = '#3b82f6'
        rate_label = 'Slow'
        rate_desc = 'Below average speaking speed'
    else:
        rate_color = '#10b981'
        rate_label = 'Normal'
        rate_desc = 'Average conversational pace'
    
    output = f"""
<div style="background: linear-gradient(135deg, #eff6ff 0%, white 100%); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid #3b82f633;">
    <h3 style="margin: 0; color: #1e40af; font-size: 22px; font-weight: 700;">Speech Transcription</h3>
    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 14px;">Detected language: <strong>{asr['language'].upper()}</strong></p>
</div>

<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Speech Metrics</h4>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
        <div style="background: #f9fafb; border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 24px; font-weight: 700; color: #1e40af;">{asr['duration']:.1f}</div>
            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">Duration (sec)</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 24px; font-weight: 700; color: #1e40af;">{asr['word_count']}</div>
            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">Words</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 24px; font-weight: 700; color: {rate_color};">{asr['speech_rate']:.0f}</div>
            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">Words/min</div>
        </div>
        <div style="background: {rate_color}15; border-radius: 10px; padding: 16px; text-align: center; border: 1px solid {rate_color}33;">
            <div style="font-size: 18px; font-weight: 700; color: {rate_color};">{rate_label}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">{rate_desc}</div>
        </div>
    </div>
</div>

<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Full Transcription</h4>
    <div style="background: #f9fafb; border-radius: 10px; padding: 20px; border-left: 4px solid #3b82f6;">
        <p style="margin: 0; font-size: 15px; line-height: 1.8; color: #374151; font-style: italic;">"{asr['transcription']}"</p>
    </div>
</div>
"""
    
    return output


# build the speech patterns display HTML
def build_speech_patterns_display(asr):
    output = ""
    
    if 'kopparapu_score' in asr:
        classification = asr['kopparapu_classification'].upper()
        kop_score = asr['kopparapu_score']
        confidence = kop_score if kop_score >= 0.5 else (1 - kop_score)
        
        if classification == 'SPONTANEOUS':
            class_color = '#10b981'
            class_bg = '#ecfdf5'
        else:
            class_color = '#f59e0b'
            class_bg = '#fffbeb'
        
        kf = asr['kopparapu_features']
        
        cpw = kf['chars_per_word']
        cpw_interp = 'Complex vocabulary' if cpw > 5.5 else 'Simple vocabulary' if cpw < 4.5 else 'Average'
        
        wps = kf['words_per_sec']
        wps_interp = 'Fast pace' if wps > 3 else 'Slow pace' if wps < 2 else 'Normal pace'
        
        fr = kf['filler_rate'] * 100
        fr_interp = 'High (spontaneous)' if fr > 5 else 'Low (scripted)' if fr < 2 else 'Moderate'
        
        rep = kf['repetition_count']
        rep_interp = 'Multiple (thinking aloud)' if rep > 3 else 'None (prepared)' if rep == 0 else 'Few'
        
        output += f"""
<div style="background: linear-gradient(135deg, {class_bg} 0%, white 100%); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid {class_color}33;">
    <h3 style="margin: 0; color: {class_color}; font-size: 22px; font-weight: 700;">{classification} SPEECH</h3>
    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 14px;">Linguistic analysis confidence: <strong>{confidence*100:.1f}%</strong></p>
    <div style="margin-top: 12px; background: #e5e7eb; border-radius: 6px; overflow: hidden; height: 8px;">
        <div style="height: 100%; width: {kop_score*100:.0f}%; background: linear-gradient(to right, #10b981, #f59e0b); border-radius: 6px;"></div>
    </div>
</div>

<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600;">Linguistic Metrics</h4>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: #374151;">{cpw:.2f}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Chars/Word</div>
            <div style="font-size: 10px; color: #9ca3af; margin-top: 2px;">{cpw_interp}</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: #374151;">{wps:.2f}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Words/Sec</div>
            <div style="font-size: 10px; color: #9ca3af; margin-top: 2px;">{wps_interp}</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: #374151;">{fr:.1f}%</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Filler Rate</div>
            <div style="font-size: 10px; color: #9ca3af; margin-top: 2px;">{fr_interp}</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: #374151;">{rep}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Repetitions</div>
            <div style="font-size: 10px; color: #9ca3af; margin-top: 2px;">{rep_interp}</div>
        </div>
    </div>
</div>
"""
    
    # filler words section
    filler_ratio = asr['filler_words']['ratio']
    filler_count = asr['filler_words']['count']
    
    if filler_ratio > 0.05:
        filler_color = '#10b981'
        filler_label = 'High filler usage'
        filler_desc = 'Strong indicator of spontaneous speech'
    elif filler_ratio < 0.02:
        filler_color = '#f59e0b'
        filler_label = 'Low filler usage'
        filler_desc = 'May indicate reading or rehearsed speech'
    else:
        filler_color = '#6b7280'
        filler_label = 'Moderate filler usage'
        filler_desc = 'Normal conversational pattern'
    
    output += f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600;">Filler Words</h4>
    <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 16px; align-items: center;">
        <div style="background: #f9fafb; border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 28px; font-weight: 700; color: {filler_color};">{filler_count}</div>
            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">Filler Words</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 28px; font-weight: 700; color: {filler_color};">{filler_ratio*100:.1f}%</div>
            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">Of Speech</div>
        </div>
        <div style="background: {filler_color}10; border-radius: 10px; padding: 16px; border: 1px solid {filler_color}33;">
            <div style="font-weight: 600; color: {filler_color}; font-size: 14px;">{filler_label}</div>
            <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">{filler_desc}</div>
        </div>
    </div>
</div>
"""
    
    # pause patterns section
    pause_var = asr['pause_patterns']['pause_variability']
    
    if pause_var < 0.3:
        pause_color = '#f59e0b'
        pause_label = 'Regular pauses'
        pause_desc = 'Suggests reading at punctuation marks'
    elif pause_var > 0.6:
        pause_color = '#10b981'
        pause_label = 'Irregular pauses'
        pause_desc = 'Natural thinking breaks indicate spontaneous speech'
    else:
        pause_color = '#6b7280'
        pause_label = 'Moderate variability'
        pause_desc = 'Mixed pattern'
    
    output += f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px;">
    <h4 style="margin: 0 0 16px 0; color: #374151; font-size: 15px; font-weight: 600;">Pause Patterns</h4>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: #374151;">{asr['pause_patterns']['num_pauses']}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Total Pauses</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: #374151;">{asr['pause_patterns']['avg_pause']:.2f}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Avg Duration</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: #374151;">{asr['pause_patterns']['max_pause']:.2f}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Longest Pause</div>
        </div>
        <div style="background: #f9fafb; border-radius: 10px; padding: 14px; text-align: center;">
            <div style="font-size: 20px; font-weight: 700; color: {pause_color};">{pause_var:.2f}</div>
            <div style="font-size: 11px; color: #6b7280; margin-top: 4px;">Variability</div>
        </div>
    </div>
    <div style="background: {pause_color}10; border-radius: 10px; padding: 14px; border: 1px solid {pause_color}33;">
        <div style="font-weight: 600; color: {pause_color}; font-size: 14px;">{pause_label}</div>
        <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">{pause_desc}</div>
    </div>
</div>
"""
    
    return output


# build the AI detection display HTML
def build_ai_detection_display(text_auth):
    is_ai = text_auth['ai_detection']['ai_generated']
    ai_prob = text_auth['ai_detection']['confidence']
    human_prob = 1 - ai_prob
    
    if is_ai:
        main_color = '#ef4444'
        bg_color = '#fef2f2'
        label = 'AI-GENERATED LIKELY'
        desc = 'The text shows patterns consistent with AI-generated content'
    else:
        main_color = '#10b981'
        bg_color = '#ecfdf5'
        label = 'HUMAN-WRITTEN LIKELY'
        desc = 'The text shows patterns consistent with human-written content'
    
    output = f"""
<div style="background: linear-gradient(135deg, {bg_color} 0%, white 100%); border-radius: 16px; padding: 24px; margin-bottom: 20px; border: 1px solid {main_color}33;">
    <h3 style="margin: 0; color: {main_color}; font-size: 22px; font-weight: 700;">{label}</h3>
    <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 14px;">{desc}</p>
</div>

<div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
    <h4 style="margin: 0 0 20px 0; color: #374151; font-size: 15px; font-weight: 600;">Confidence Analysis</h4>
    
    <div style="margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #ef4444; font-size: 14px;">AI Generated</span>
            <span style="font-weight: 700; color: #ef4444; font-size: 18px;">{ai_prob*100:.0f}%</span>
        </div>
        <div style="background: #fee2e2; border-radius: 8px; overflow: hidden; height: 12px;">
            <div style="height: 100%; width: {ai_prob*100:.0f}%; background: #ef4444; border-radius: 8px;"></div>
        </div>
    </div>
    
    <div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #10b981; font-size: 14px;">Human Written</span>
            <span style="font-weight: 700; color: #10b981; font-size: 18px;">{human_prob*100:.0f}%</span>
        </div>
        <div style="background: #d1fae5; border-radius: 8px; overflow: hidden; height: 12px;">
            <div style="height: 100%; width: {human_prob*100:.0f}%; background: #10b981; border-radius: 8px;"></div>
        </div>
    </div>
</div>

<div style="background: #fffbeb; border: 1px solid #fcd34d; border-radius: 10px; padding: 14px;">
    <div style="font-size: 13px; color: #92400e; line-height: 1.5;">
        <strong>Note:</strong> AI detection is probabilistic and should be used as one factor among many in your evaluation.
    </div>
</div>
"""
    
    return output


def analyze_audio_file(audio_file):
    # check if pipeline is ready
    if not pipeline_ready:
        error_msg = pipeline_error if 'pipeline_error' in dir() else "Something went wrong"
        error_html = f"""
<div style="background: #fef2f2; border: 1px solid #ef4444; border-radius: 12px; padding: 20px;">
    <h3 style="margin: 0 0 8px 0; color: #dc2626; font-size: 16px;">Pipeline not ready</h3>
    <p style="margin: 0; color: #7f1d1d; font-size: 14px;">{error_msg}</p>
</div>
"""
        return (error_html, "", "", "", "")
    
    # check if audio file was provided
    if audio_file is None:
        placeholder_html = """
<div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 40px; text-align: center;">
    <p style="margin: 0; color: #6b7280; font-size: 15px;">Please upload an audio file to begin analysis.</p>
</div>
"""
        return (placeholder_html, "", "", "", "")
    
    # run analysis
    try:
        language_code = None
        results = pipeline.analyze_audio(audio_file, language=language_code)
        
        # extract results from each component
        audio_class = results['audio_classification']
        asr = results['speech_recognition']
        text_auth = results['text_authenticity']
        final = results['final_assessment']
        
        verdict_color = {
            "AUTHENTIC": "#10b981",
            "LIKELY AUTHENTIC": "#3b82f6",
            "QUESTIONABLE": "#f59e0b",
            "LIKELY INAUTHENTIC": "#ef4444"
        }
        
        color = verdict_color.get(final['verdict'], '#6b7280')
        
        overall_status = f"""
<div style='background: white; border: 2px solid {color}; padding: 25px; border-radius: 16px; margin: 10px 0;'>
    <h2 style='color: {color}; margin: 0 0 15px 0; font-size: 24px; font-weight: 700;'>
        {final['verdict']}
    </h2>
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 15px 0;'>
        <div style='text-align: center; padding: 15px; background: white; border-radius: 10px;'>
            <div style='font-size: 2em; font-weight: bold; color: {color};'>{final['composite_authenticity_score']*100:.0f}%</div>
            <div style='color: #666; margin-top: 5px;'>Authenticity Score</div>
        </div>
        <div style='text-align: center; padding: 15px; background: white; border-radius: 10px;'>
            <div style='font-size: 2em; font-weight: bold; color: {color};'>{final['risk_level'].upper()}</div>
            <div style='color: #666; margin-top: 5px;'>Risk Level</div>
        </div>
        <div style='text-align: center; padding: 15px; background: white; border-radius: 10px;'>
            <div style='font-size: 2em; font-weight: bold; color: #667eea;'>{results['processing_time']:.1f}s</div>
            <div style='color: #666; margin-top: 5px;'>Processing Time</div>
        </div>
    </div>
    <div style='background: white; padding: 15px; border-radius: 10px; margin-top: 15px;'>
        <em style='color: #555;'>{final['recommendation']}</em>
    </div>
</div>
"""
        # build tab outputs
        acoustic_output = build_acoustic_features_display(audio_class)
        transcription_output = build_transcription_display(asr)
        speech_patterns = build_speech_patterns_display(asr)
        ai_output = build_ai_detection_display(text_auth)
                
        return (
            overall_status,
            acoustic_output,
            transcription_output,
            speech_patterns,
            ai_output,
        )
        
    except Exception as e:
        error_html = f"""
<div style="background: #fef2f2; border: 1px solid #ef4444; border-radius: 12px; padding: 20px;">
    <h3 style="margin: 0 0 12px 0; color: #dc2626; font-size: 16px;">Something went wrong</h3>
    <p style="margin: 0 0 12px 0; color: #7f1d1d; font-size: 14px;">{str(e)}</p>
    <details style="margin-top: 12px;">
        <summary style="color: #6b7280; cursor: pointer; font-size: 13px;">More info</summary>
        <pre style="background: #1f2937; color: #f3f4f6; padding: 12px; border-radius: 8px; margin-top: 8px; font-size: 11px; overflow-x: auto;">{traceback.format_exc()}</pre>
    </details>
</div>
"""
        return (error_html, "", "", "", "")


# create the gradio interface
def create_interface():    
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
    
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif !important;
        background: white !important;
    }
    button.primary, .primary {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
    }
    """
    
    with gr.Blocks(title="Authenticity Detection System") as demo:
        
        gr.HTML(f"""
        <style>
        {custom_css}
        </style>
        <header style='background: white; border-bottom: 1px solid #e5e7eb; margin-bottom: 32px;'>
            <div style='padding: 16px 0;'>
                <div style='display: flex; align-items: center; gap: 12px;'>
                    <div>
                        <p style='margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #6b7280; font-weight: 500;'>
                            LEIDEN UNIVERSITY · LIACS
                        </p>
                        <h1 style='margin: 0; font-size: 18px; font-weight: 600; color: #111827;'>
                            Audio Processing & Indexing Project
                        </h1>
                    </div>
                </div>
            </div>
        </header>
        
        <section style='background: linear-gradient(to bottom, white, #f9fafb); margin-bottom: 40px;'>
            <div style='padding: 32px 0;'>
                <h2 style='font-size: 32px; font-weight: 700; line-height: 1.2; color: #111827; margin: 0 0 16px 0;'>
                    Detecting AI-Assisted Responses in Online Settings
                </h2>
                <div style='display: flex; flex-wrap: wrap; gap: 12px;'>
                    <span style='display: inline-flex; align-items: center; padding: 8px 16px; background: #eff6ff; color: #1e40af; border-radius: 8px; font-size: 14px; font-weight: 500;'>
                        Multi-Modal Analysis
                    </span>
                    <span style='display: inline-flex; align-items: center; padding: 8px 16px; background: #fef3c7; color: #92400e; border-radius: 8px; font-size: 14px; font-weight: 500;'>
                        Acoustic + Linguistic
                    </span>
                </div>
            </div>
        </section>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style='background: white; border: 1px solid #e5e7eb; padding: 20px; border-radius: 16px; margin-bottom: 20px;'>
                    <h3 style='margin: 0; font-size: 18px; font-weight: 600; color: #111827;'>Audio Input</h3>
                    <p style='margin: 8px 0 0 0; font-size: 14px; color: #6b7280;'>Upload or record your audio file</p>
                </div>
                """)
                
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio File",
                    show_label=False
                )
                
                analyze_btn = gr.Button(
                    "Analyze Audio", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.HTML("""
                <div style='background: white; border: 1px solid #e5e7eb; padding: 20px; border-radius: 16px; margin-top: 20px;'>
                    <h4 style='margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: #111827;'>Requirements</h4>
                    <ul style='margin: 0; padding-left: 20px; font-size: 13px; color: #6b7280; line-height: 1.8;'>
                        <li><strong>Formats:</strong> WAV, MP3, M4A, FLAC, OGG</li>
                        <li><strong>Duration:</strong> 30 sec - 5 min</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.HTML("""
                <div style='background: white; border: 1px solid #e5e7eb; padding: 20px; border-radius: 16px; margin-bottom: 20px;'>
                    <h3 style='margin: 0; font-size: 18px; font-weight: 600; color: #111827;'>Analysis Results</h3>
                    <p style='margin: 8px 0 0 0; font-size: 14px; color: #6b7280;'>You'll see results here</p>
                </div>
                """)
                
                overall_output = gr.HTML()
                
                # results tabs
                with gr.Tabs() as tabs:
                    with gr.Tab("Acoustic Features"):
                        acoustic_output = gr.HTML()
                    
                    with gr.Tab("Transcription"):
                        transcription_output = gr.HTML()
                    
                    with gr.Tab("Speech Patterns"):
                        speech_output = gr.HTML()
                    
                    with gr.Tab("AI Detection"):
                        ai_output = gr.HTML()
        
        # example audio files
        gr.HTML("""
        <div style='margin-top: 20px; margin-bottom: 10px;'>
            <h4 style='margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #111827;'>Try these examples:</h4>
        </div>
        """)
        
        examples_dir = os.path.join(os.path.dirname(__file__), "examples")
        gr.Examples(
            examples=[
                [os.path.join(examples_dir, "read1.wav")],
                [os.path.join(examples_dir, "spontaneous1.wav")]
            ],
            inputs=[audio_input],
            outputs=[
                overall_output,
                acoustic_output,
                transcription_output,
                speech_output,
                ai_output,
            ],
            fn=analyze_audio_file,
            label="",
            examples_per_page=2,
            cache_examples=False
        )
        
        def show_loading():
            loading_html = """
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: 2px solid #667eea; padding: 30px; border-radius: 16px; margin: 10px 0; text-align: center;'>
    <h2 style='color: white; margin: 0 0 15px 0; font-size: 24px; font-weight: 700;'>
        Analyzing...
    </h2>
</div>
"""
            loading_tab = """
<div style='padding: 40px; text-align: center; color: #6b7280;'>
    <p style='margin-top: 16px; font-size: 14px;'>Processing...</p>
</div>
"""
            return loading_html, loading_tab, loading_tab, loading_tab, loading_tab
        
        analyze_btn.click(
            fn=show_loading,
            inputs=None,
            outputs=[
                overall_output,
                acoustic_output,
                transcription_output,
                speech_output,
                ai_output,
            ],
            queue=False
        ).then(
            fn=analyze_audio_file,
            inputs=[audio_input],
            outputs=[
                overall_output,
                acoustic_output,
                transcription_output,
                speech_output,
                ai_output,
            ]
        )
        
        # footer
        gr.HTML("""
        <footer style='border-top: 1px solid #e5e7eb; background: white; margin-top: 48px; padding: 32px 0;'>
            <div style='text-align: center;'>
                <p style='margin: 0; font-size: 14px; color: #6b7280;'>
                </p>
            </div>
        </footer>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
