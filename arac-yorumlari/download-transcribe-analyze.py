import os
import mlx_whisper
import ollama
import httpx
from tqdm import tqdm
import time
import yt_dlp
import subprocess
import re

# --- AYARLAR ---
MODEL_NAME = "qwen2.5:32b" 
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
OUTPUT_HTML = "index.html"
TXT_FOLDER = "desifreler"

def sanitize_filename(filename):
    """Video başlığındaki işletim sistemi için yasaklı karakterleri temizler."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()

def get_all_videos(url_input):
    """URL'yi derinlemesine tarar ve playlistteki tüm videoları eksiksiz listeler."""
    print(f"\n[🔍] URL Kaynağı taranıyor: {url_input}")
    print("ℹ️  Geniş oynatma listelerinde videoların taranması 1 dakika kadar sürebilir...")
    
    ydl_opts = {
        'extract_flat': 'in_playlist',
        'skip_download': True,
        'quiet': True,
        'nocheckcertificate': True,
        'playlist_items': '1-1000'
    }
    
    video_list = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url_input, download=False)
            
            if 'entries' in info_dict:
                entries = list(info_dict['entries'])
                print(f"📂 Bağlantıda bir oynatma listesi/kanal tespit edildi. İçerik eşleniyor...")
                for entry in entries:
                    if entry:
                        video_list.append({
                            'url': f"https://www.youtube.com/watch?v={entry['id']}",
                            'title': entry.get('title', 'Bilinmeyen Video Title')
                        })
            else:
                video_list.append({
                    'url': url_input,
                    'title': info_dict.get('title', 'Bilinmeyen Video Title')
                })
        return video_list
    except Exception as e:
        print(f"❌ '{url_input}' kaynağından video listesi alınırken hata oluştu: {e}")
        return []

def download_with_terminal_subproces(video_url):
    """Homebrew altındaki bağımsız yt-dlp'yi verbose olarak tetikler."""
    output_filename = "temp_video_file"
    
    cmd = [
        "/opt/homebrew/bin/yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-x", "--audio-format", "mp3", "--audio-quality", "0",
        "--embed-thumbnail",
        "--add-metadata",
        "-o", f"{output_filename}.%(ext)s",
        "--quiet",
        "--cookies-from-browser=safari",
        video_url
    ]
    
    try:
        print(f"   ↳ [📥 İndirme] yt-dlp terminal komutu koşturuluyor...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        expected_mp3 = f"{output_filename}.mp3"
        if os.path.exists(expected_mp3):
            print(f"   ↳ [📥 İndirme] Başarılı! Dosya hazır: {expected_mp3}")
            return expected_mp3
        else:
            if result.stderr:
                print(f"   ↳ ❌ yt-dlp Hatası: {result.stderr.strip()}")
            return None
    except Exception as e:
        print(f"   ↳ ⚠️ Subprocess hatası: {e}")
        return None

def transcribe_audio(file_path):
    """Whisper deşifre adımını detaylı raporlar."""
    if not os.path.exists(file_path):
        return ""
    print(f"   ↳ [🎙️ Deşifre] Whisper ({WHISPER_MODEL}) GPU üzerinde motoru ateşledi...")
    start_t = time.time()
    
    result = mlx_whisper.transcribe(file_path, path_or_hf_repo=WHISPER_MODEL)
    full_text = " ".join([seg['text'].strip() for seg in result['segments'] if seg['text'].strip()])
    
    print(f"   ↳ [🎙️ Deşifre] Tamamlandı. Süre: {int(time.time() - start_t)} sn. Metin boyutu: {len(full_text)} karakter.")
    return full_text

def chunk_text_by_words(text, max_words=800):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def call_ollama_with_timeout(model, prompt, timeout=1200, max_retries=3):
    """Timeout ve retry ile Ollama çağrısı yapar. Donma sorununu önler."""
    for attempt in range(max_retries):
        try:
            client = ollama.Client(timeout=httpx.Timeout(timeout))
            response = client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": 0.0}
            )
            return response['response'].strip()
        except Exception as e:
            print(f"   ↳ ⚠️ Ollama timeout/hata (deneme {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   ↳ 🔄 10sn bekleyip tekrar deneniyor...")
                time.sleep(10)
    return ""

def extract_info_with_ai(full_transcript_text, video_title, video_url):
    """Qwen 14B ile metinden araç bilgilerini ayıklar. Yıldızlar (**) ve ### işaretleri temizlenir."""
    final_data = []
    text_chunks = chunk_text_by_words(full_transcript_text, max_words=800)
    
    print(f"   ↳ [🤖 AI Analizi] Transkript {len(text_chunks)} adet büyük parçaya bölündü.")
    
    for idx, chunk in enumerate(text_chunks):
        print(f"   ↳ [🤖 AI Analizi] {idx+1}/{len(text_chunks)} numaralı parça {MODEL_NAME} modeline gönderildi...")
        
        prompt = (
            f"SİSTEM TALİMATI: Çince karakter (中文) kullanmak KESİNLİKLE yasaktır. "
            f"Yazdığın her bir kelime saf Türkçe olmalıdır.\n\n"
            f"GÖREV: Aşağıdaki otomobil inceleme metnini oku. Metinde bahsedilen tüm arabaları bul. Araçların özellikle sorunları hakkındaki yorumlarda spesifik bilgiyi al ve kaydet.\n"
            f"Her araba için şu 4 bilgiyi alt alta şablon halinde yaz:\n"
            f"Marka:\n"
            f"Model:\n"
            f"Yıl:\n"
            f"Yorum:\n\n"
            f"Metin:\n{chunk}\n"
        )
        
        try:
            start_ai = time.time()
            response = call_ollama_with_timeout(MODEL_NAME, prompt, timeout=180)
            if not response:
                print(f"   ↳ ❌ Parça {idx+1} 3 denemede de cevap alınamadı, atlanıyor.")
                continue
            
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            current_item = {"marka": "", "model": "", "yil": "", "yorum": ""}
            last_key = None
            local_added = 0
            
            for line in lines:
                l_line = line.lower().lstrip("*-•0123456789. ")
                
                if l_line.startswith("marka:"):
                    if current_item["marka"] and current_item["yorum"]:
                        current_item["video_baslik"] = video_title
                        current_item["video_url"] = video_url
                        final_data.append(current_item.copy())
                        local_added += 1
                        current_item = {"marka": "", "model": "", "yil": "", "yorum": ""}
                    
                    content = line[line.lower().find("marka:") + 6:].strip().replace("**", "")
                    if content: current_item["marka"] = content
                    else: last_key = "marka"
                    
                elif l_line.startswith("model:"):
                    content = line[line.lower().find("model:") + 6:].strip().replace("**", "")
                    if content: current_item["model"] = content
                    else: last_key = "model"
                    
                elif l_line.startswith("yıl:") or l_line.startswith("yil:"):
                    idx_key = line.lower().find("yıl:") if "yıl:" in line.lower() else line.lower().find("yil:")
                    content = line[idx_key + 4:].strip().replace("**", "")
                    if content: current_item["yil"] = content
                    else: last_key = "yil"
                    
                elif l_line.startswith("yorum:"):
                    content = line[line.lower().find("yorum:") + 6:].strip().replace("**", "")
                    if content: current_item["yorum"] = content
                    else: last_key = "yorum"
                else:
                    if last_key:
                        current_item[last_key] = line.replace("**", "")
                        last_key = None
                    elif current_item["yorum"]:
                        if "###" in line:
                            line = line.split("###")[0].strip()
                        
                        line = line.replace("**", "")
                        if line:
                            current_item["yorum"] += " " + line

            if current_item["marka"] and current_item["yorum"]:
                current_item["video_baslik"] = video_title
                current_item["video_url"] = video_url
                final_data.append(current_item)
                local_added += 1
                
            print(f"   ↳ [🤖 AI Analizi] Parça {idx+1} bitti ({int(time.time() - start_ai)} sn). {local_added} araç yakalandı.")
                        
        except Exception as e:
            print(f"   ↳ [🤖 AI Analizi] ⚠️ Parça {idx+1} işlenirken hata atlandı: {e}")
            continue
            
    return final_data

def get_already_processed_urls():
    """index.html dosyasını hızlıca tarayıp daha önce işlenmiş tüm video linklerini küme (set) olarak döner."""
    processed_urls = set()
    if not os.path.exists(OUTPUT_HTML):
        return processed_urls
    try:
        with open(OUTPUT_HTML, "r", encoding="utf-8") as f:
            content = f.read()
        urls = re.findall(r"href='(https://www\.youtube\.com/watch\?v=.*?)'", content)
        for url in urls:
            processed_urls.add(url.strip())
    except Exception:
        pass
    return processed_urls

def load_existing_html_data():
    """Eski index.html dosyasını okur ve en soldaki yeni sıra numarası düzenine göre kayıtları kurtarır."""
    existing_items = []
    if not os.path.exists(OUTPUT_HTML):
        return existing_items
    try:
        with open(OUTPUT_HTML, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Kritik Güncelleme: Sıra numarası (No) tagi regex'in en başına (<tr> arkasına) taşındı.
        pattern = r"<tr>\s*<td><span class='text-muted'>\d+</span></td>\s*<td><span class='badge bg-danger text-uppercase filter-clickable' style='cursor: pointer;'>(.*?)</span></td>\s*<td><b class='filter-clickable' style='cursor: pointer; color: #0d6efd;'>(.*?)</b></td>\s*<td><span class='text-muted'>(.*?)</span></td>\s*<td>(.*?)</td>\s*<td><a href='(.*?)' target='_blank' class='badge bg-secondary text-wrap' style='text-decoration: none;'>(.*?)</a></td>\s*</tr>"
        rx_rows = re.findall(pattern, content, re.DOTALL)
        for row in rx_rows:
            existing_items.append({
                "marka": row[0].strip(),
                "model": row[1].strip(),
                "yil": row[2].strip(),
                "yorum": row[3].strip(),
                "video_url": row[4].strip(),
                "video_baslik": row[5].strip()
            })
    except Exception:
        pass
    return existing_items

def generate_html_report(new_data):
    """Eski verileri korur, sıra numaralarını en solda hesaplar ve HTML raporu yazar."""
    all_data = load_existing_html_data()
    seen_signatures = {f"{d['marka'].lower()}-{d['model'].lower()}-{d['yil']}-{d['yorum'][:20].lower()}" for d in all_data}
    
    added_count = 0
    for nd in new_data:
        sig = f"{nd['marka'].lower()}-{nd['model'].lower()}-{nd['yil']}-{nd['yorum'][:20].lower()}"
        if sig not in seen_signatures:
            all_data.append(nd)
            seen_signatures.add(sig)
            added_count += 1
            
    unique_videos = {d['video_url'] for d in all_data if d.get('video_url')}
    total_videos_count = len(unique_videos)
    
    ai_rows = ""
    for idx, d in enumerate(all_data, 1):
        url_target = d.get('video_url', '#')
        title_target = d.get('video_baslik', 'Bilinmeyen Video')
        
        # Sıra numarası <td> hücresi tablonun EN BAŞINA çekildi
        ai_rows += f"""
        <tr>
            <td><span class='text-muted'>{idx}</span></td>
            <td><span class='badge bg-danger text-uppercase filter-clickable' style='cursor: pointer;'>{d['marka']}</span></td>
            <td><b class='filter-clickable' style='cursor: pointer; color: #0d6efd;'>{d['model']}</b></td>
            <td><span class='text-muted'>{d['yil']}</span></td>
            <td>{d['yorum']}</td>
            <td><a href='{url_target}' target='_blank' class='badge bg-secondary text-wrap' style='text-decoration: none;'>{title_target}</a></td>
        </tr>"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Otomobil İnceleme Ansiklopedisi</title>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #fcfcfc; padding: 40px; font-family: 'Segoe UI', system-ui, sans-serif; }}
            .main-card {{ background: white; border: none; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.04); padding: 30px; }}
            h2 {{ font-weight: 800; color: #111; }}
            a.badge:hover {{ background-color: #0d6efd !important; color: white !important; transition: 0.2s; }}
            .filter-clickable:hover {{ text-decoration: underline; opacity: 0.8; }}
            #clearFilterBtn {{ display: none; }}
            .stats-container {{ margin-bottom: 25px; }}
            .stat-badge {{ font-size: 0.95rem; padding: 8px 16px; border-radius: 30px; font-weight: 600; display: inline-block; margin: 0 5px; }}
        </style>
    </head>
    <body>
        <div class="container-fluid">
            <div class="main-card">
                <h2 class="text-center text-primary mb-3">🚗 ARABA DEDEKTİFİ OTOMOBİL VERİTABANI</h2>
                
                <div class="text-center stats-container">
                    <span class="stat-badge bg-light text-dark shadow-sm border">
                        Mevcut toplam analiz edilen araç sayısı: <b class="text-danger" style="font-size: 1.1rem;">{len(all_data)}</b>
                    </span>
                    <span class="stat-badge bg-light text-dark shadow-sm border">
                        Toplam analiz edilmiş video sayısı: <b class="text-primary" style="font-size: 1.1rem;">{total_videos_count}</b>
                    </span>
                </div>
                
                <div class="mb-3 text-end">
                    <button id="clearFilterBtn" class="btn btn-warning btn-sm shadow-sm">🧹 Filtreyi Temizle (<span id="activeFilterText"></span>)</button>
                </div>

                <table id="aiTable" class="table table-hover align-middle">
                    <thead class="table-dark">
                        <tr>
                            <th>No</th><th>Marka</th><th>Model</th><th>Yıl</th><th>Öneri ve Kronik Sorunlar Özeti</th><th>Kaynak Video Adı (Tıklanabilir)</th>
                        </tr>
                    </thead>
                    <tbody>{ai_rows}</tbody>
                </table>
            </div>
        </div>
        <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script>
            $(document).ready(function() {{
                var table = $('#aiTable').DataTable({{ 
                    "language": {{"url": "//cdn.datatables.net/plug-ins/1.13.6/i18n/tr.json"}}, 
                    "pageLength": 25, 
                    "order": [[ 0, "asc" ]] // İlk sütun olan 'No' baz alınarak kronolojik sıralanır
                }});
                $('#aiTable tbody').on('click', '.filter-clickable', function() {{
                    var cellText = $(this).text().trim();
                    table.search(cellText).draw();
                    $('#activeFilterText').text(cellText);
                    $('#clearFilterBtn').fadeIn(200);
                }});
                $('#clearFilterBtn').on('click', function() {{
                    table.search('').draw();
                    $(this).fadeOut(200);
                }});
            }});
        </script>
    </body>
    </html>
    """
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   ↳ [💾 Rapor Kayıt] index.html dosyası başarıyla güncellendi (Flush tamam).")

def main():
    print("--- 🏁 ARABA DEDEKTİFİ v27 (En Sol Sıra Numaralı Tasarım) 🏁 ---")
    raw_input_urls = input("Video/Playlist linklerini VIRGÜLLE AYIRARAK girin:\n👉 ").strip()
    
    os.makedirs(TXT_FOLDER, exist_ok=True)
    
    target_urls = [url.strip() for url in raw_input_urls.split(",") if url.strip()]
    
    all_videos_pool = []
    print(f"\n[🔄] Girilen {len(target_urls)} kaynak link arka planda analiz ediliyor...")
    
    for target_url in target_urls:
        playlist_videos = get_all_videos(target_url)
        all_videos_pool.extend(playlist_videos)
        
    if not all_videos_pool:
        print("❌ İşlenecek hiçbir video bulunamadı. Program kapatılıyor.")
        return
        
    print("\n" + "="*60)
    print("📋 SİSTEM TARAFINDAN TESPİT EDİLEN TÜM VİDEOLARIN LİSTESİ:")
    print("="*60)
    for v_idx, video in enumerate(all_videos_pool):
        print(f"[{v_idx+1}] {video['title']} \n    ↳ Link: {video['url']}")
    print("="*60)
    print(f"🔍 Toplam {len(all_videos_pool)} adet video işleme sırasına alındı.\n")
    
    start_time = time.time()
    
    # Dev video havuzunu sırayla işlemeye başla
    for idx, video in enumerate(all_videos_pool):
        print(f"\n🎬 [{idx+1}/{len(all_videos_pool)}] İŞLEM SIRASI: {video['title']}")
        
        live_processed_urls = get_already_processed_urls()
        
        if video['url'] in live_processed_urls:
            print(f"   🟢 [BYPASS] Bu video veritabanında zaten mevcut. Sıradakine geçiliyor...")
            time.sleep(0.05)
            continue
            
        print(f"   🚀 [YENİ ANALİZ] Bu video ilk kez işlenecek. Süreç tetiklendi.")
        audio_file = download_with_terminal_subproces(video['url'])
        
        if not audio_file or not os.path.exists(audio_file):
            print(f"   ⚠️ Video dosyasına erişilemedi (Members only / Hata), sıradaki video taranacak.")
            continue
            
        try:
            transcript = transcribe_audio(file_path=audio_file)
            
            if transcript:
                safe_title = sanitize_filename(video['title'])
                txt_filename = os.path.join(TXT_FOLDER, f"{safe_title}.txt")
                
                with open(txt_filename, "w", encoding="utf-8") as txt_file:
                    txt_file.write(transcript)
                print(f"   ↳ [📄 TXT Kayıt] Ham deşifre metni arşivlendi: {txt_filename}")
                
                video_extracted_data = extract_info_with_ai(transcript, video['title'], video['url'])
                generate_html_report(video_extracted_data)
                
        except Exception as e:
            print(f"   ❌ Kritik döngü hatası: {e}")
        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
            thumb_file = audio_file.replace(".mp3", ".jpg")
            if os.path.exists(thumb_file):
                os.remove(thumb_file)
                
    print(f"\n🎉 GECE OTOMASYONU KUSURSUZ ŞEKİLDE TAMAMLANDI!")
    print(f"⏱️ Toplam Çalışma Süresi: {int((time.time() - start_time) / 60)} dakika.")

if __name__ == "__main__":
    main()