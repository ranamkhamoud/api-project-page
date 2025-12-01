import gradio as gr
import os
from pipeline import AuthenticityDetectionPipeline
import traceback

try:
    pipeline = AuthenticityDetectionPipeline(whisper_model_size="base")
    pipeline_ready = True
except Exception:
    pipeline_ready = False


def analyze_audio_file(audio_file):
    if not pipeline_ready:
        return (
            "Error: Pipeline not initialized. Please check the installation.",
            "", "", "", ""
        )
    
    if audio_file is None:
        return (
            "Please upload an audio file.",
            "", "", "", ""
        )
    
    try:
        language_code = None
        results = pipeline.analyze_audio(audio_file, language=language_code)
        
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
        acoustic_output = audio_class['interpretation']
        
        transcription_output = "### Speech Transcription\n\n"
        transcription_output += f"| Metric | Value |\n"
        transcription_output += f"|--------|-------|\n"
        transcription_output += f"| **Language** | {asr['language'].upper()} |\n"
        transcription_output += f"| **Duration** | {asr['duration']:.1f} seconds |\n"
        transcription_output += f"| **Word Count** | {asr['word_count']} words |\n"
        transcription_output += f"| **Speech Rate** | {asr['speech_rate']:.1f} words/min |\n\n"
        if asr['speech_rate'] > 160:
            transcription_output += "**Fast speech rate** - Above average speaking speed\n\n"
        elif asr['speech_rate'] < 120:
            transcription_output += "**Slow speech rate** - Below average speaking speed\n\n"
        else:
            transcription_output += "**Normal speech rate** - Average conversational pace\n\n"
        
        transcription_output += "---\n\n"
        transcription_output += "#### Full Transcription\n\n"
        transcription_output += f"> {asr['transcription']}"
        
        
        if 'kopparapu_score' in asr:
            classification = asr['kopparapu_classification'].upper()
            confidence = asr['kopparapu_score'] if asr['kopparapu_score'] >= 0.5 else (1 - asr['kopparapu_score'])
            
            speech_patterns = f" ### **Classification: {classification} SPEECH**\n\n"
            speech_patterns += f"**Score:** {asr['kopparapu_score']:.3f} (0=spontaneous, 1=read)\n"
            speech_patterns += f"**Confidence:** {confidence*100:.1f}%\n\n"
            
            speech_patterns += "---\n\n"
            speech_patterns += "#### Linguistic Metrics\n\n"
            kf = asr['kopparapu_features']
            
            speech_patterns += "| Feature | Value | Interpretation |\n"
            speech_patterns += "|---------|-------|----------------|\n"
            speech_patterns += f"| **Characters/Word** | {kf['chars_per_word']:.2f} | "
            if kf['chars_per_word'] > 5.5:
                speech_patterns += "Complex vocabulary |\n"
            elif kf['chars_per_word'] < 4.5:
                speech_patterns += "Simple vocabulary |\n"
            else:
                speech_patterns += "Average complexity |\n"
            
            speech_patterns += f"| **Words/Second** | {kf['words_per_sec']:.2f} | "
            if kf['words_per_sec'] > 3:
                speech_patterns += "Fast pacing |\n"
            elif kf['words_per_sec'] < 2:
                speech_patterns += "Slow pacing |\n"
            else:
                speech_patterns += "Normal pacing |\n"
            
            speech_patterns += f"| **Non-alpha chars/sec** | {kf['nonalpha_per_sec']:.2f} | "
            if kf['nonalpha_per_sec'] > 2.5:
                speech_patterns += "High (disfluent) |\n"
            elif kf['nonalpha_per_sec'] < 1.5:
                speech_patterns += "Low (fluent) |\n"
            else:
                speech_patterns += "Moderate |\n"
            
            speech_patterns += f"| **Filler Rate** | {kf['filler_rate']*100:.1f}% | "
            if kf['filler_rate'] > 0.05:
                speech_patterns += "High (spontaneous) |\n"
            elif kf['filler_rate'] < 0.02:
                speech_patterns += "Low (scripted) |\n"
            else:
                speech_patterns += "Moderate |\n"
            
            speech_patterns += f"| **Repetitions** | {kf['repetition_count']} | "
            if kf['repetition_count'] > 3:
                speech_patterns += "Multiple (thinking aloud) |\n"
            elif kf['repetition_count'] == 0:
                speech_patterns += "None (prepared) |\n"
            else:
                speech_patterns += "Few |\n"
            
            speech_patterns += f"| **Alpha Ratio** | {kf['alpha_ratio']:.2f} | "
            if kf['alpha_ratio'] > 0.85:
                speech_patterns += "Clean text |\n"
            else:
                speech_patterns += "With artifacts |\n"
            
            speech_patterns += "\n"
        
        speech_patterns += "---\n\n"
        speech_patterns += "#### Filler Words & Disfluencies\n\n"
        filler_ratio = asr['filler_words']['ratio']
        speech_patterns += f"**Count:** {asr['filler_words']['count']} filler words\n"
        speech_patterns += f"**Ratio:** {filler_ratio*100:.2f}% of speech\n\n"
        
        if asr['filler_words']['details']:
            speech_patterns += "**Found:** " + ', '.join([f"*{k}* ({v})" for k, v in asr['filler_words']['details'].items()]) + "\n\n"
        
        if filler_ratio > 0.05:
            speech_patterns += "**High filler usage** - Strong indicator of spontaneous, unscripted speech\n\n"
        elif filler_ratio < 0.02:
            speech_patterns += "**Low filler usage** - May indicate reading or highly rehearsed speech\n\n"
        else:
            speech_patterns += "**Moderate filler usage** - Normal conversational pattern\n\n"
        
        speech_patterns += "---\n\n"
        speech_patterns += "#### Pause Patterns\n\n"
        pause_var = asr['pause_patterns']['pause_variability']
        
        speech_patterns += f"**Total Pauses:** {asr['pause_patterns']['num_pauses']}\n"
        speech_patterns += f"**Average Duration:** {asr['pause_patterns']['avg_pause']:.2f}s\n"
        speech_patterns += f"**Longest Pause:** {asr['pause_patterns']['max_pause']:.2f}s\n"
        speech_patterns += f"**Variability:** {pause_var:.2f}\n\n"
        
        if pause_var < 0.3:
            speech_patterns += "**Regular pauses** - Consistent pattern suggests reading at punctuation marks\n\n"
        elif pause_var > 0.6:
            speech_patterns += "**Irregular pauses** - Natural thinking breaks indicate spontaneous speech\n\n"
        else:
            speech_patterns += "**Moderate variability** - Mixed pattern\n\n"
        
        is_ai = text_auth['ai_detection']['ai_generated']
        ai_prob = text_auth['ai_detection']['confidence']
        
        if is_ai:
            ai_output = "### **AI-GENERATED LIKELY**\n\n"
        else:
            ai_output = "### **HUMAN-WRITTEN LIKELY**\n\n"
        
        ai_output += "**Confidence:**\n\n"
        bar_length = 30
        ai_bars = int(ai_prob * bar_length)
        human_bars = bar_length - ai_bars
        ai_output += f"```\nAI:    [{'█' * ai_bars}{'░' * human_bars}] {ai_prob*100:.0f}%\n"
        ai_output += f"Human: [{'█' * human_bars}{'░' * ai_bars}] {(1-ai_prob)*100:.0f}%\n```\n\n"
        
        ai_output += "---\n\n"
        ai_output += "#### Interpretation\n\n"
        ai_interpretation = text_auth['ai_detection'].get('interpretation', 'No interpretation available.')
        if ai_interpretation:
            ai_output += ai_interpretation
        else:
            ai_output += "No interpretation available."
                
        return (
            overall_status,
            acoustic_output,
            transcription_output,
            speech_patterns,
            ai_output,
        )
        
    except Exception as e:
        error_msg = f"Error during analysis:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return (error_msg, "", "", "", "", "")


def create_interface():    
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
    
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif !important;
        background: white !important;
    }
    .contain {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 auto !important;
        background: white !important;
        padding: 0 !important;
    }
    .tab-nav button {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 14px;
        font-weight: 500;
        padding: 10px 16px;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s;
    }
    .tab-nav button.selected {
        background: #2563eb;
        color: white;
        font-weight: 600;
    }
    button.primary, .primary {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
    }
    button.primary:hover, .primary:hover {
        background: #1d4ed8 !important;
    }
    .markdown-text {
        font-family: 'IBM Plex Sans', sans-serif;
        line-height: 1.7;
    }
    h1, h2, h3, h4 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
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
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="32" height="32">
                        <defs>
                            <linearGradient id="g" x1="0" y1="0" x2="64" y2="0" gradientUnits="userSpaceOnUse">
                                <stop offset="0" stop-color="#1d4ed8" />
                                <stop offset="1" stop-color="#0ea5e9" />
                            </linearGradient>
                        </defs>
                        <rect x="0" y="0" width="64" height="64" rx="12" fill="#ffffff"/>
                        <path d="M4 32 C 10 18, 18 46, 24 32 S 36 18, 40 32 52 46, 60 32"
                              fill="none" stroke="url(#g)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
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
                <p style='font-size: 18px; color: #374151; margin: 0 0 24px 0;'>
                </p>
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
                <div style='background: white; border: 1px solid #e5e7eb; padding: 20px; border-radius: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;'>
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
                
                # Add example audio files
                gr.HTML("""
                <div style='margin-top: 20px; margin-bottom: 10px;'>
                    <h4 style='margin: 0 0 8px 0; font-size: 14px; font-weight: 600; color: #111827;'>Try these examples:</h4>
                </div>
                """)
                
                examples_dir = os.path.join(os.path.dirname(__file__), "examples")
                gr.Examples(
                    examples=[
                        [os.path.join(examples_dir, "read1.ogg")],
                        [os.path.join(examples_dir, "spontaneous1.ogg")]
                    ],
                    inputs=[audio_input],
                    label="",
                    examples_per_page=2,
                    cache_examples=False
                )
                
                gr.HTML("""
                <div style='background: white; border: 1px solid #e5e7eb; padding: 20px; border-radius: 16px; margin-top: 20px;'>
                    <h4 style='margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: #111827;'>Requirements</h4>
                    <ul style='margin: 0; padding-left: 20px; font-size: 13px; color: #6b7280; line-height: 1.8;'>
                        <li><strong>Formats:</strong> WAV, MP3, M4A, FLAC, OGG</li>
                        <li><strong>Duration:</strong> 30 sec - 5 min</li>
                    </ul>
                </div>
                
                <div style='background: #fef3c7; border: 1px solid #fbbf24; padding: 16px; border-radius: 12px; margin-top: 16px;'>
                    <div style='font-size: 12px; color: #92400e; line-height: 1.6;'>
                        <strong>Note:</strong> Provides probabilistic assessments. 
                        Use as one factor in evaluation.
                    </div>
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.HTML("""
                <div style='background: white; border: 1px solid #e5e7eb; padding: 20px; border-radius: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                    <h3 style='margin: 0; font-size: 18px; font-weight: 600; color: #111827;'>Analysis Results</h3>
                    <p style='margin: 8px 0 0 0; font-size: 14px; color: #6b7280;'>You'll see results here</p>
                </div>
                """)
                
                overall_output = gr.Markdown()
                
                with gr.Tabs() as tabs:
                    with gr.Tab("Acoustic Features"):
                        acoustic_output = gr.Markdown()
                    
                    with gr.Tab("Transcription"):
                        transcription_output = gr.Markdown()
                    
                    with gr.Tab("Speech Patterns"):
                        speech_output = gr.Markdown()
                    
                    with gr.Tab("AI Detection"):
                        ai_output = gr.Markdown()
                    

        
        def show_loading():
            loading_html = """
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: 2px solid #667eea; padding: 30px; border-radius: 16px; margin: 10px 0; text-align: center;'>
    <h2 style='color: white; margin: 0 0 15px 0; font-size: 24px; font-weight: 700;'>
        Analyzing...
    </h2>
    <div style='margin-top: 20px;'>
        <div style='display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: white; margin: 0 4px; animation: pulse 1.5s ease-in-out infinite;'></div>
        <div style='display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: white; margin: 0 4px; animation: pulse 1.5s ease-in-out 0.2s infinite;'></div>
        <div style='display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: white; margin: 0 4px; animation: pulse 1.5s ease-in-out 0.4s infinite;'></div>
    </div>
</div>
<style>
@keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(0.8); }
    50% { opacity: 1; transform: scale(1.2); }
}
</style>
"""
            loading_msg = " **Processing...**"
            return loading_html, loading_msg, loading_msg, loading_msg, loading_msg
        
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
        
        gr.HTML("""
        <footer style='border-top: 1px solid #e5e7eb; background: white; margin-top: 48px; padding: 32px 0;'>
            <div style='text-align: center;'>
                <p style='margin: 0; font-size: 14px; color: #6b7280;'>
                </p>
                <p style='margin: 8px 0 0 0; font-size: 13px; color: #9ca3af;'>
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

