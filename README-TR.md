<div align="center">

# 🤖 Free Claude Code

### Claude Code CLI ve VSCode'u ücretsiz kullanın. Anthropic API anahtarı gerekmez.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/uv)
[![Pytest ile Test Edildi](https://img.shields.io/badge/testing-Pytest-00c0ff.svg?style=for-the-badge)](https://github.com/Alishahryar1/free-claude-code/actions/workflows/tests.yml)
[![Tip Kontrolü: Ty](https://img.shields.io/badge/type%20checking-ty-ffcc00.svg?style=for-the-badge)](https://pypi.org/project/ty/)
[![Kod Stili: Ruff](https://img.shields.io/badge/code%20formatting-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Loglama: Loguru](https://img.shields.io/badge/logging-loguru-4ecdc4.svg?style=for-the-badge)](https://github.com/Delgan/loguru)

Claude Code'un Anthropic API isteklerini **NVIDIA NIM** (dakikada 40 ücretsiz istek), **OpenRouter** (yüzlerce model), **DeepSeek** (doğrudan API), **LM Studio** (tamamen yerel), **llama.cpp** (Anthropic endpoint'li yerel) veya **Ollama**'ya (tamamen yerel, native Anthropic Messages) yönlendiren hafif bir proxy.

[Hızlı Başlangıç](#hızlı-başlangıç) · [Sağlayıcılar](#sağlayıcılar) · [Discord Botu](#discord-botu) · [Yapılandırma](#yapılandırma) · [Geliştirme](#geliştirme) · [Katkıda Bulunma](#katkıda-bulunma)

---

</div>

<div align="center">
  <img src="pic.png" alt="Free Claude Code çalışırken" width="700">
  <p><em>NVIDIA NIM üzerinden çalışan Claude Code — tamamen ücretsiz</em></p>
</div>

## Özellikler

| Özellik                        | Açıklama                                                                                                              |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **Sıfır Maliyet**              | NVIDIA NIM'de dakikada 40 ücretsiz istek. OpenRouter'da ücretsiz modeller. LM Studio, Ollama veya llama.cpp ile tamamen yerel |
| **Doğrudan Yerine Geçer**      | 2 ortam değişkeni ayarla, bitti. Claude Code CLI veya VSCode eklentisinde hiçbir değişiklik gerekmez                 |
| **6 Sağlayıcı**                | NVIDIA NIM, OpenRouter, DeepSeek, LM Studio (yerel), llama.cpp (`llama-server`), Ollama                              |
| **Model Bazlı Yönlendirme**    | Opus / Sonnet / Haiku isteklerini farklı model ve sağlayıcılara yönlendir. Sağlayıcıları özgürce karıştır            |
| **Thinking Token Desteği**     | `<think>` etiketleri ve `reasoning_content` alanı, native Claude thinking bloklarına dönüştürülür                    |
| **Buluşsal Araç Ayrıştırıcı** | Araç çağrılarını metin olarak döndüren modeller otomatik olarak yapılandırılmış araç kullanımına ayrıştırılır        |
| **İstek Optimizasyonu**        | 5 kategorideki önemsiz API çağrısı yerel olarak karşılanır; kota ve gecikme tasarrufu sağlar                         |
| **Akıllı Hız Sınırlama**       | Proaktif kayan pencere kısıtlaması + reaktif 429 üstel geri çekilme + isteğe bağlı eşzamanlılık sınırı              |
| **Discord / Telegram Botu**    | Ağaç tabanlı iş parçacığı, oturum kalıcılığı ve canlı ilerleme ile uzaktan otonom kodlama                           |
| **Alt Ajan Kontrolü**          | Task aracına müdahale ile `run_in_background=False` zorlanır. Kontrolden çıkan alt ajan yok                          |
| **Genişletilebilir**           | Temiz `BaseProvider` ve `MessagingPlatform` soyut sınıfları. Yeni sağlayıcı veya platform eklemek kolaydır          |

## Hızlı Başlangıç

### Ön Koşullar

1. Bir API anahtarı edinin (veya yerel bir sağlayıcı kullanın):
   - **NVIDIA NIM**: [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
   - **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys)
   - **DeepSeek**: [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
   - **LM Studio**: API anahtarı gerekmez. [LM Studio](https://lmstudio.ai) ile yerel olarak çalıştırın
   - **llama.cpp**: API anahtarı gerekmez. `llama-server`'ı yerel olarak çalıştırın.
   - **Ollama**: API anahtarı gerekmez. [Ollama](https://ollama.com) ile yerel olarak çalıştırın (`ollama serve`).
2. [Claude Code](https://github.com/anthropics/claude-code)'u kurun

### `uv` Kurulumu

```bash
# Önerilen yükleyici (macOS/Linux için, sistem pip'ine bağımlı değil)
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv zaten kuruluysa güncelleyin
uv self update

# Bu proje Python 3.14 gerektirir
uv python install 3.14
```

PowerShell (Windows):

```powershell
# Önerilen yükleyici (sistem pip'ine bağımlı olmaz)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# uv zaten kuruluysa güncelleyin
uv self update

# Bu proje Python 3.14 gerektirir
uv python install 3.14
```

> **Not:** Homebrew ile yönetilen Python'da `pip install uv` komutu, PEP 668 nedeniyle `externally-managed-environment` hatasıyla başarısız olabilir. Yukarıdaki resmi yükleyiciyi tercih edin.

### Klonla ve Yapılandır

```bash
git clone https://github.com/Alishahryar1/free-claude-code.git
cd free-claude-code
cp .env.example .env
```

Sağlayıcınızı seçin ve `.env` dosyasını düzenleyin:

<details>
<summary><b>NVIDIA NIM</b> (dakikada 40 ücretsiz istek, önerilir)</summary>

```dotenv
NVIDIA_NIM_API_KEY="nvapi-your-key-here"

MODEL_OPUS=
MODEL_SONNET=
MODEL_HAIKU=
MODEL="nvidia_nim/z-ai/glm4.7"                     # yedek

# Model başına düşünme isteği ve Claude thinking blokları için anahtarlar.
# Boş bırakılan model anahtarları ENABLE_MODEL_THINKING değerini miras alır.
ENABLE_OPUS_THINKING=
ENABLE_SONNET_THINKING=
ENABLE_HAIKU_THINKING=
ENABLE_MODEL_THINKING=true
```

</details>

<details>
<summary><b>OpenRouter</b> (yüzlerce model)</summary>

```dotenv
OPENROUTER_API_KEY="sk-or-your-key-here"

MODEL_OPUS="open_router/deepseek/deepseek-r1-0528:free"
MODEL_SONNET="open_router/openai/gpt-oss-120b:free"
MODEL_HAIKU="open_router/stepfun/step-3.5-flash:free"
MODEL="open_router/stepfun/step-3.5-flash:free"     # yedek
```

</details>

<details>
<summary><b>DeepSeek</b> (doğrudan API)</summary>

```dotenv
DEEPSEEK_API_KEY="your-deepseek-key-here"

MODEL_SONNET="deepseek/deepseek-chat"
MODEL_HAIKU="deepseek/deepseek-chat"
MODEL="deepseek/deepseek-chat"                      # yedek
```

</details>

<details>
<summary><b>LM Studio</b> (tamamen yerel, API anahtarı gerekmez)</summary>

```dotenv
MODEL_OPUS="lmstudio/unsloth/MiniMax-M2.5-GGUF"
MODEL_SONNET="lmstudio/unsloth/Qwen3.5-35B-A3B-GGUF"
MODEL_HAIKU="lmstudio/unsloth/GLM-4.7-Flash-GGUF"
MODEL="lmstudio/unsloth/GLM-4.7-Flash-GGUF"         # yedek
```

</details>

<details>
<summary><b>llama.cpp</b> (tamamen yerel, API anahtarı gerekmez)</summary>

```dotenv
LLAMACPP_BASE_URL="http://localhost:8080/v1"

MODEL_OPUS="llamacpp/local-model"
MODEL_SONNET="llamacpp/local-model"
MODEL_HAIKU="llamacpp/local-model"
MODEL="llamacpp/local-model"
```

</details>

<details>
<summary><b>Ollama</b> (tamamen yerel, API anahtarı gerekmez)</summary>

```dotenv
OLLAMA_BASE_URL="http://localhost:11434"

MODEL_OPUS="ollama/llama3.1"
MODEL_SONNET="ollama/llama3.1"
MODEL_HAIKU="ollama/llama3.1"
MODEL="ollama/llama3.1"                             # yedek
```

Kurulum: [ollama.com](https://ollama.com). Bir model indirin (`ollama pull llama3.1`) ve sunucuyu çalışır durumda tutun (`ollama serve` veya masaüstü uygulaması). `MODEL*` alanlarında `ollama list` çıktısındaki model etiketiyle aynı değeri kullanın (örneğin `ollama/llama3.1:8b`).

</details>

<details>
<summary><b>Sağlayıcıları Karıştır</b></summary>

Her `MODEL_*` değişkeni farklı bir sağlayıcı kullanabilir. `MODEL`, tanınmayan Claude modelleri için yedektir.

```dotenv
NVIDIA_NIM_API_KEY="nvapi-your-key-here"
OPENROUTER_API_KEY="sk-or-your-key-here"

MODEL_OPUS="nvidia_nim/moonshotai/kimi-k2.5"
MODEL_SONNET="open_router/deepseek/deepseek-r1-0528:free"
MODEL_HAIKU="lmstudio/unsloth/GLM-4.7-Flash-GGUF"
MODEL="nvidia_nim/z-ai/glm4.7"                      # yedek
```

</details>

> **Geçiş Notu:** `NIM_ENABLE_THINKING` ve `ENABLE_THINKING` bu sürümde kaldırıldı. Yedek anahtar olarak `ENABLE_MODEL_THINKING`; isteğe bağlı geçersiz kılma için `ENABLE_OPUS_THINKING`, `ENABLE_SONNET_THINKING` ve `ENABLE_HAIKU_THINKING` kullanın.

<details>
<summary><b>İsteğe Bağlı Kimlik Doğrulama</b> (proxy erişimini kısıtla)</summary>

Proxy'ye erişimi kısıtlamak için `.env` dosyasına `ANTHROPIC_AUTH_TOKEN` ekleyin:

```dotenv
ANTHROPIC_AUTH_TOKEN="your-secret-token-here"
```

**Nasıl çalışır:**
- `ANTHROPIC_AUTH_TOKEN` boşsa (varsayılan) kimlik doğrulama istenmez (geriye dönük uyumlu)
- Ayarlandığında, istemciler aynı token'ı `ANTHROPIC_AUTH_TOKEN` başlığıyla sağlamalıdır
- `claude-pick` betiği, yapılandırılmışsa token'ı `.env`'den otomatik okur

**Örnek kullanım:**
```bash
# Kimlik doğrulamayla
ANTHROPIC_AUTH_TOKEN="your-secret-token-here" \
ANTHROPIC_BASE_URL="http://localhost:8082" claude

# claude-pick yapılandırılmış token'ı otomatik kullanır
claude-pick
```

Bu özelliği şu durumlarda kullanın:
- Proxy'yi genel bir ağda çalıştırıyorsanız
- Sunucuyu başkalarıyla paylaşıyor ama erişimi kısıtlamak istiyorsanız
- Ek bir güvenlik katmanı istiyorsanız

</details>

### Çalıştır

**Terminal 1:** Proxy sunucusunu başlatın:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2:** Claude Code'u çalıştırın:

`ANTHROPIC_BASE_URL`'i proxy kök URL'sine yönlendirin; `http://localhost:8082/v1` değil, `http://localhost:8082` kullanın.

#### PowerShell
```powershell
$env:ANTHROPIC_AUTH_TOKEN="freecc"; $env:ANTHROPIC_BASE_URL="http://localhost:8082"; claude
```
#### Bash
```bash
ANTHROPIC_AUTH_TOKEN="freecc" ANTHROPIC_BASE_URL="http://localhost:8082" claude
```

Hepsi bu kadar! Claude Code artık yapılandırdığınız sağlayıcıyı ücretsiz olarak kullanıyor.

<details>
<summary><b>VSCode Eklenti Kurulumu</b></summary>

1. Proxy sunucusunu başlatın (yukarıdaki gibi).
2. Ayarları açın (`Ctrl + ,`) ve `claude-code.environmentVariables` arayın.
3. **settings.json'da Düzenle**'ye tıklayın ve şunu ekleyin:

```json
"claudeCode.environmentVariables": [
  { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082" },
  { "name": "ANTHROPIC_AUTH_TOKEN", "value": "freecc" }
]
```

4. Eklentileri yeniden yükleyin.
5. **Giriş ekranı görünürse**: **Anthropic Console**'a tıklayın ve yetkilendirin. Eklenti çalışmaya başlayacaktır. Tarayıcıda kredi satın alma sayfasına yönlendirilebilirsiniz; bunu görmezden gelin — eklenti zaten çalışıyor.

Anthropic modellerine geri dönmek için eklenen bloğu yorum satırına alın ve eklentileri yeniden yükleyin.

</details>

<details>
<summary><b>IntelliJ Eklenti Kurulumu</b></summary>

1. Yapılandırma dosyasını açın:
   - **Windows**: `C:\Users\%USERNAME%\AppData\Roaming\JetBrains\acp-agents\installed.json`
   - **Linux/macOS**: `~/.jetbrains/acp.json`
2. `acp.registry.claude-acp` içinde şunu değiştirin:

   ```
   "env": {}
   ```
   bununla:

   ```
   "env": {
   "ANTHROPIC_AUTH_TOKEN": "freecc",
   "ANTHROPIC_BASE_URL": "http://localhost:8082"
   }
   ```
3. Proxy sunucusunu başlatın
4. IDE'yi yeniden başlatın

</details>

<details>
<summary><b>Çoklu Model Desteği (Model Seçici)</b></summary>

`claude-pick`, Claude'u her başlatışınızda `.env` dosyasındaki `MODEL`'i düzenlemeden aktif sağlayıcıdan herhangi bir model seçmenizi sağlayan etkileşimli bir model seçicidir.

https://github.com/user-attachments/assets/9a33c316-90f8-4418-9650-97e7d33ad645

**1. [fzf](https://github.com/junegunn/fzf)'yi yükleyin**:

```bash
brew install fzf        # macOS/Linux
```

**2. `~/.zshrc` veya `~/.bashrc`'ye alias ekleyin:**

```bash
alias claude-pick="/free-claude-code/claude-pick'in/tam/yolu"
```

Ardından kabuğunuzu yeniden yükleyin (`source ~/.zshrc` veya `source ~/.bashrc`) ve `claude-pick` çalıştırın.

**Ya da sabit model alias'ı kullanın** (seçici gerekmez):

```bash
alias claude-kimi='ANTHROPIC_BASE_URL="http://localhost:8082" ANTHROPIC_AUTH_TOKEN="freecc:moonshotai/kimi-k2.5" claude'
```

</details>

### Paket Olarak Kur (klonlama gerekmez)

```bash
uv tool install git+https://github.com/Alishahryar1/free-claude-code.git
fcc-init        # yerleşik şablondan ~/.config/free-claude-code/.env oluşturur
```

`~/.config/free-claude-code/.env` dosyasını API anahtarlarınız ve model adlarıyla düzenleyin, ardından:

```bash
free-claude-code    # sunucuyu başlatır
```

> Güncellemek için: `uv tool upgrade free-claude-code`

---

## Nasıl Çalışır

```
┌─────────────────┐        ┌──────────────────────┐        ┌──────────────────┐
│  Claude Code    │───────>│  Free Claude Code    │───────>│  LLM Sağlayıcı   │
│  CLI / VSCode   │<───────│  Proxy (:8082)       │<───────│  NIM / OR / LMS  │
└─────────────────┘        └──────────────────────┘        └──────────────────┘
   Anthropic API                                             Native Anthropic
   formatı (SSE)                                            veya OpenAI chat SSE
```

- **Şeffaf proxy**: Claude Code standart Anthropic API istekleri gönderir; proxy bunları yapılandırılmış sağlayıcınıza iletir
- **Model bazlı yönlendirme**: Opus / Sonnet / Haiku istekleri model özelindeki arka uca çözümlenir; yedek olarak `MODEL` kullanılır
- **İstek optimizasyonu**: 5 kategorideki önemsiz istek (kota yoklaması, başlık üretimi, önek tespiti, öneri modu, dosya yolu çıkarımı) API kotası kullanılmadan yerel olarak karşılanır
- **Format dönüşümü**: OpenRouter, LM Studio, llama.cpp ve Ollama native Anthropic Messages endpoint'lerini kullanır; NIM ve DeepSeek ise paylaşılan OpenAI chat çevirisiyle çalışır
- **Thinking token'ları**: Çözümlenen modelin thinking anahtarı etkinleştirildiğinde `<think>` etiketleri ve `reasoning_content` alanları native Claude thinking bloklarına dönüştürülür

Proxy ayrıca Claude uyumlu yoklama rotalarını sunar: `GET /v1/models`, `POST /v1/messages`, `POST /v1/messages/count_tokens` ve yaygın yoklama endpoint'leri için `HEAD`/`OPTIONS` desteği.

---

## Sağlayıcılar

| Sağlayıcı      | Maliyet      | Hız Sınırı    | En Uygun Kullanım                              |
| -------------- | ------------ | ------------- | ---------------------------------------------- |
| **NVIDIA NIM** | Ücretsiz     | 40 istek/dak  | Günlük kullanım, cömert ücretsiz katman         |
| **OpenRouter** | Ücretsiz/Ücretli | Değişken  | Model çeşitliliği, yedek seçenekler            |
| **DeepSeek**   | Kullanım bazlı | Değişken   | DeepSeek chat/reasoner'a doğrudan erişim       |
| **LM Studio**  | Ücretsiz (yerel) | Sınırsız | Gizlilik, çevrimdışı kullanım, hız sınırı yok |
| **llama.cpp**  | Ücretsiz (yerel) | Sınırsız | Hafif yerel çıkarım motoru                     |
| **Ollama**     | Ücretsiz (yerel) | Sınırsız | Kolay yerel LLM çalışma zamanı, native Anthropic API |

Modeller `sağlayıcı_öneki/model/adı` formatını kullanır. Geçersiz bir önek hataya yol açar.

| Sağlayıcı   | `MODEL` Öneki     | API Anahtar Değişkeni | Varsayılan Temel URL          |
| ----------- | ----------------- | --------------------- | ----------------------------- |
| NVIDIA NIM  | `nvidia_nim/...`  | `NVIDIA_NIM_API_KEY`  | `integrate.api.nvidia.com/v1` |
| OpenRouter  | `open_router/...` | `OPENROUTER_API_KEY`  | `openrouter.ai/api/v1`        |
| DeepSeek    | `deepseek/...`    | `DEEPSEEK_API_KEY`    | `api.deepseek.com`            |
| LM Studio   | `lmstudio/...`    | (yok)                 | `localhost:1234/v1`           |
| llama.cpp   | `llamacpp/...`    | (yok)                 | `localhost:8080/v1`           |
| Ollama      | `ollama/...`      | (yok)                 | `localhost:11434`             |

<details>
<summary><b>NVIDIA NIM Modelleri</b></summary>

Popüler modeller (tam liste [`nvidia_nim_models.json`](nvidia_nim_models.json) dosyasında):

- `nvidia_nim/minimaxai/minimax-m2.5`
- `nvidia_nim/qwen/qwen3.5-397b-a17b`
- `nvidia_nim/z-ai/glm5`
- `nvidia_nim/moonshotai/kimi-k2.5`
- `nvidia_nim/stepfun-ai/step-3.5-flash`

Göz atın: [build.nvidia.com](https://build.nvidia.com/explore/discover) · Listeyi güncelleyin: `curl "https://integrate.api.nvidia.com/v1/models" > nvidia_nim_models.json`

</details>

<details>
<summary><b>OpenRouter Modelleri</b></summary>

Popüler ücretsiz modeller:

- `open_router/arcee-ai/trinity-large-preview:free`
- `open_router/stepfun/step-3.5-flash:free`
- `open_router/deepseek/deepseek-r1-0528:free`
- `open_router/openai/gpt-oss-120b:free`

Göz atın: [openrouter.ai/models](https://openrouter.ai/models) · [Ücretsiz modeller](https://openrouter.ai/collections/free-models)

</details>

<details>
<summary><b>DeepSeek Modelleri</b></summary>

DeepSeek şu anda şu doğrudan API modellerini sunmaktadır:

- `deepseek/deepseek-chat`
- `deepseek/deepseek-reasoner`

Göz atın: [api-docs.deepseek.com](https://api-docs.deepseek.com)

</details>

<details>
<summary><b>LM Studio Modelleri</b></summary>

[LM Studio](https://lmstudio.ai) ile modelleri yerel olarak çalıştırın. Chat veya Developer sekmesinde bir model yükleyin, ardından `MODEL`'i model tanımlayıcısına ayarlayın.

Native araç kullanım desteğiyle örnekler:

- `LiquidAI/LFM2-24B-A2B-GGUF`
- `unsloth/MiniMax-M2.5-GGUF`
- `unsloth/GLM-4.7-Flash-GGUF`
- `unsloth/Qwen3.5-35B-A3B-GGUF`

Göz atın: [model.lmstudio.ai](https://model.lmstudio.ai)

</details>

<details>
<summary><b>llama.cpp Modelleri</b></summary>

`llama-server` kullanarak modelleri yerel olarak çalıştırın. Araç yetenekli bir GGUF dosyanız olduğundan emin olun. `llama-server`, `/v1/messages` üzerinden çalışırken model adını yok saydığından `MODEL`'e istediğiniz bir ad verebilirsiniz (örn. `llamacpp/my-model`).

Ayrıntılı talimatlar ve uyumlu modeller için Unsloth belgelerine bakın:
[https://unsloth.ai/docs/models/qwen3.5](https://unsloth.ai/docs/models/qwen3.5)

</details>

<details>
<summary><b>Ollama Modelleri</b></summary>

[Ollama](https://ollama.com) ile modelleri yerel olarak çalıştırın. Bir model indirin, ardından `MODEL`'i `ollama/<etiket>` biçiminde ayarlayın; `<etiket>`, `ollama list` çıktısındaki adla eşleşmelidir (örneğin `ollama/llama3.1:8b` veya `ollama/qwen2.5-coder:7b`).

- `OLLAMA_BASE_URL`, **Ollama sunucu kökü**dür (varsayılan `http://localhost:11434`). `/v1` eklemeyin; proxy, Ollama'nın native Anthropic Messages desteğini o host üzerinden kullanır.
- Ollama farklı bir adres veya portu dinliyorsa `OLLAMA_BASE_URL`'i geçersiz kılın.

```bash
ollama pull llama3.1
ollama serve   # veya sunucuyu çalışır tutan masaüstü uygulamasını kullanın
```

Göz atın: [ollama.com/library](https://ollama.com/library)

</details>

---

## Discord Botu

Claude Code'u Discord'dan (veya Telegram'dan) uzaktan kontrol edin. Görevler gönderin, canlı ilerlemeyi izleyin ve birden fazla eşzamanlı oturumu yönetin.

**Yetenekler:**

- Ağaç tabanlı mesaj iş parçacığı: bir mesajı yanıtlayarak konuşmayı dallara ayırın
- Sunucu yeniden başlatmaları arasında oturum kalıcılığı
- Thinking token'ları, araç çağrıları ve sonuçların canlı akışı
- Sınırsız eşzamanlı Claude CLI oturumu (`PROVIDER_MAX_CONCURRENCY` ile kontrol edilir)
- Sesli notlar: sesli mesaj gönderin; bunlar yazıya dönüştürülerek normal prompt olarak işlenir
- Komutlar: `/stop` (görevi iptal et; sadece o görevi durdurmak için bir mesajı yanıtlayın), `/clear` (tüm oturumları sıfırla veya bir dalı temizlemek için yanıtlayın), `/stats`

### Kurulum

1. **Discord Botu Oluşturun**: [Discord Geliştirici Portalı](https://discord.com/developers/applications)'na gidin, bir uygulama oluşturun, bot ekleyin ve token'ı kopyalayın. Bot ayarlarında **Message Content Intent**'i etkinleştirin.

2. **`.env`'i düzenleyin:**

```dotenv
MESSAGING_PLATFORM="discord"
DISCORD_BOT_TOKEN="your_discord_bot_token"
ALLOWED_DISCORD_CHANNELS="123456789,987654321"
```

> Discord'da Geliştirici Modunu etkinleştirin (Ayarlar → Gelişmiş), ardından bir kanala sağ tıklayın ve "ID'yi Kopyala"ya basın. Birden fazla kanalı virgülle ayırın. Boş bırakılırsa hiçbir kanala izin verilmez.

3. **Çalışma alanını yapılandırın** (Claude'un işlem yapacağı dizin):

```dotenv
CLAUDE_WORKSPACE="./agent_workspace"
ALLOWED_DIR="C:/Users/kullaniciadi/projeler"
```

4. **Sunucuyu başlatın:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

5. **Botu davet edin**: OAuth2 URL Generator aracılığıyla (kapsamlar: `bot`, izinler: Mesaj Oku, Mesaj Gönder, Mesajları Yönet, Mesaj Geçmişini Oku).

### Telegram

`MESSAGING_PLATFORM=telegram` ayarlayın ve yapılandırın:

```dotenv
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
ALLOWED_TELEGRAM_USER_ID="your_telegram_user_id"
```

Token için [@BotFather](https://t.me/BotFather)'a; kullanıcı ID'si için [@userinfobot](https://t.me/userinfobot)'a başvurun.

### Sesli Notlar

Discord veya Telegram'da sesli mesaj gönderin; bunlar yazıya dönüştürülerek normal prompt olarak işlenir.

| Arka Uç                        | Açıklama                                                                                                       | API Anahtarı         |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- | -------------------- |
| **Yerel Whisper** (varsayılan) | [Hugging Face Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) — ücretsiz, çevrimdışı, CUDA uyumlu | gerekmez           |
| **NVIDIA NIM**                 | gRPC üzerinden Whisper/Parakeet modelleri                                                                      | `NVIDIA_NIM_API_KEY` |

**Ses eklentilerini kurun:**

```bash
# Repoyu klonladıysanız:
uv sync --extra voice_local          # Yerel Whisper
uv sync --extra voice                # NVIDIA NIM
uv sync --extra voice --extra voice_local  # Her ikisi

# Paket olarak kurduysanız (klonlama yok):
uv tool install "free-claude-code[voice_local] @ git+https://github.com/Alishahryar1/free-claude-code.git"
uv tool install "free-claude-code[voice] @ git+https://github.com/Alishahryar1/free-claude-code.git"
uv tool install "free-claude-code[voice,voice_local] @ git+https://github.com/Alishahryar1/free-claude-code.git"
```

`WHISPER_DEVICE` (`cpu` | `cuda` | `nvidia_nim`) ve `WHISPER_MODEL` ile yapılandırın. Tüm ses değişkenleri ve desteklenen model değerleri için [Yapılandırma](#yapılandırma) tablosuna bakın.

---

## Yapılandırma

### Temel

| Değişken             | Açıklama                                                              | Varsayılan                                        |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| `MODEL`              | Yedek model (`sağlayıcı/model/adı` formatı; geçersiz önek → hata)   | `nvidia_nim/z-ai/glm4.7`                          |
| `MODEL_OPUS`         | Claude Opus istekleri için model; boş bırakılırsa `MODEL`'e düşer    | boş                                               |
| `MODEL_SONNET`       | Claude Sonnet istekleri için model; boş bırakılırsa `MODEL`'e düşer  | boş                                               |
| `MODEL_HAIKU`        | Claude Haiku istekleri için model; boş bırakılırsa `MODEL`'e düşer   | boş                                               |
| `NVIDIA_NIM_API_KEY`    | NVIDIA API anahtarı                                                | NIM için zorunlu                                  |
| `ENABLE_MODEL_THINKING` | Sağlayıcı düşünme istekleri ve Claude thinking blokları için yedek anahtar. `false` olarak ayarlayarak model katmanı geçersiz kılmadıkça thinking'i gizleyin. | `true` |
| `ENABLE_OPUS_THINKING` | Claude Opus istekleri için isteğe bağlı thinking anahtarı; boş bırakılırsa `ENABLE_MODEL_THINKING`'i miras alır. | boş |
| `ENABLE_SONNET_THINKING` | Claude Sonnet istekleri için isteğe bağlı thinking anahtarı; boş bırakılırsa `ENABLE_MODEL_THINKING`'i miras alır. | boş |
| `ENABLE_HAIKU_THINKING` | Claude Haiku istekleri için isteğe bağlı thinking anahtarı; boş bırakılırsa `ENABLE_MODEL_THINKING`'i miras alır. | boş |
| `OPENROUTER_API_KEY` | OpenRouter API anahtarı                                               | OpenRouter için zorunlu                           |
| `DEEPSEEK_API_KEY`   | DeepSeek API anahtarı                                                 | DeepSeek için zorunlu                             |
| `LM_STUDIO_BASE_URL` | LM Studio sunucu URL'si                                               | `http://localhost:1234/v1`                        |
| `LLAMACPP_BASE_URL`  | llama.cpp sunucu URL'si                                               | `http://localhost:8080/v1`                        |
| `NVIDIA_NIM_PROXY`   | NVIDIA NIM istekleri için isteğe bağlı proxy URL'si (`http://...` veya `socks5://...`) | `""` |
| `OPENROUTER_PROXY`   | OpenRouter istekleri için isteğe bağlı proxy URL'si (`http://...` veya `socks5://...`) | `""` |
| `LMSTUDIO_PROXY`     | LM Studio istekleri için isteğe bağlı proxy URL'si (`http://...` veya `socks5://...`) | `""` |
| `LLAMACPP_PROXY`     | llama.cpp istekleri için isteğe bağlı proxy URL'si (`http://...` veya `socks5://...`) | `""` |
| `OLLAMA_BASE_URL`    | Ollama sunucu kök URL'si                                              | `http://localhost:11434`                          |

### Hız Sınırlama ve Zaman Aşımı

| Değişken                   | Açıklama                                      | Varsayılan |
| -------------------------- | --------------------------------------------- | ---------- |
| `PROVIDER_RATE_LIMIT`      | Pencere başına LLM API istek sayısı           | `40`       |
| `PROVIDER_RATE_WINDOW`     | Hız sınırı penceresi (saniye)                 | `60`       |
| `PROVIDER_MAX_CONCURRENCY` | Maksimum eşzamanlı açık sağlayıcı akışı       | `5`        |
| `HTTP_READ_TIMEOUT`        | Sağlayıcı istekleri için okuma zaman aşımı (s) | `120`      |
| `HTTP_WRITE_TIMEOUT`       | Sağlayıcı istekleri için yazma zaman aşımı (s) | `10`       |
| `HTTP_CONNECT_TIMEOUT`     | Sağlayıcı istekleri için bağlantı zaman aşımı (s) | `10`   |

### Mesajlaşma ve Ses

| Değişken                   | Açıklama                                                                                                                                                        | Varsayılan          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `MESSAGING_PLATFORM`       | `discord` veya `telegram`                                                                                                                                       | `discord`           |
| `DISCORD_BOT_TOKEN`        | Discord bot token'ı                                                                                                                                             | `""`                |
| `ALLOWED_DISCORD_CHANNELS` | Virgülle ayrılmış kanal ID'leri (boş = izin verilmez)                                                                                                          | `""`                |
| `TELEGRAM_BOT_TOKEN`       | Telegram bot token'ı                                                                                                                                            | `""`                |
| `ALLOWED_TELEGRAM_USER_ID` | İzin verilen Telegram kullanıcı ID'si                                                                                                                           | `""`                |
| `CLAUDE_WORKSPACE`         | Ajanın işlem yaptığı dizin                                                                                                                                      | `./agent_workspace` |
| `ALLOWED_DIR`              | Ajan için izin verilen dizinler                                                                                                                                 | `""`                |
| `MESSAGING_RATE_LIMIT`     | Pencere başına mesajlaşma mesaj sayısı                                                                                                                          | `1`                 |
| `MESSAGING_RATE_WINDOW`    | Mesajlaşma penceresi (saniye)                                                                                                                                   | `1`                 |
| `VOICE_NOTE_ENABLED`       | Sesli not işlemeyi etkinleştir                                                                                                                                  | `true`              |
| `WHISPER_DEVICE`           | `cpu` \| `cuda` \| `nvidia_nim`                                                                                                                                 | `cpu`               |
| `WHISPER_MODEL`            | Whisper modeli (yerel: `tiny`/`base`/`small`/`medium`/`large-v2`/`large-v3`/`large-v3-turbo`; NIM: `openai/whisper-large-v3`, `nvidia/parakeet-ctc-1.1b-asr` vb.) | `base`           |
| `HF_TOKEN`                 | Daha hızlı indirme için Hugging Face token'ı (yerel Whisper, isteğe bağlı)                                                                                     | —                   |

<details>
<summary><b>Gelişmiş: İstek optimizasyon bayrakları</b></summary>

Bu bayraklar varsayılan olarak etkindir ve API kotasından tasarruf etmek için önemsiz Claude Code isteklerini yerel olarak karşılar.

| Değişken                          | Açıklama                              | Varsayılan |
| --------------------------------- | ------------------------------------- | ---------- |
| `FAST_PREFIX_DETECTION`           | Hızlı önek tespitini etkinleştir     | `true`     |
| `ENABLE_NETWORK_PROBE_MOCK`       | Ağ yoklama isteklerini taklit et     | `true`     |
| `ENABLE_TITLE_GENERATION_SKIP`    | Başlık oluşturma isteklerini atla    | `true`     |
| `ENABLE_SUGGESTION_MODE_SKIP`     | Öneri modu isteklerini atla          | `true`     |
| `ENABLE_FILEPATH_EXTRACTION_MOCK` | Dosya yolu çıkarımını taklit et      | `true`     |

</details>

Tüm desteklenen parametreler için [`.env.example`](.env.example) dosyasına bakın.

---

## Geliştirme

### Proje Yapısı

```
free-claude-code/
├── server.py              # Giriş noktası
├── api/                   # FastAPI rotaları, API servis katmanı, model yönlendirme, istek tespiti, optimizasyonlar
├── core/                  # Paylaşılan Anthropic protokol yardımcıları, SSE, dönüşüm, ayrıştırıcılar, token sayımı
├── providers/             # Sağlayıcı kaydı, kapsamlı çalışma zamanı durumu, OpenAI chat + Anthropic messages transportları
├── messaging/             # MessagingPlatform soyut sınıfı + Discord/Telegram botları, komutlar, ses, oturum yönetimi
├── config/                # Ayarlar, NIM yapılandırması, loglama
├── cli/                   # CLI oturumu ve süreç yönetimi
└── tests/                 # Pytest test paketi
```

### Komutlar

```bash
uv run ruff format     # Kodu formatla
uv run ruff check      # Lint kontrolü
uv run ty check        # Tip kontrolü
uv run pytest          # Testleri çalıştır
```

### Genişletme

**OpenAI uyumlu sağlayıcı ekleme** (Groq, Together AI vb.) — `OpenAIChatTransport`'u genişletin, ardından sağlayıcı kaydına bir tanımlayıcı ekleyin:

```python
from providers.openai_compat import OpenAIChatTransport
from providers.base import ProviderConfig

class MyProvider(OpenAIChatTransport):
    def __init__(self, config: ProviderConfig):
        super().__init__(config, provider_name="MYPROVIDER",
                         base_url="https://api.example.com/v1", api_key=config.api_key)
```

**Native Anthropic sağlayıcısı ekleme** — `AnthropicMessagesTransport`'u genişletin, ardından `providers.registry`'ye bir tanımlayıcı ekleyin.

**Tamamen özel sağlayıcı ekleme** — `BaseProvider`'ı doğrudan genişletin, `stream_response()`'u uygulayın ve tanımlayıcısını kaydedin.

**Mesajlaşma platformu ekleme** — `messaging/` içinde `MessagingPlatform`'u genişletin ve `start()`, `stop()`, `send_message()`, `edit_message()` ile `on_message()`'i uygulayın.

---

## Katkıda Bulunma

- Hata bildirin veya özellik önerin: [Issues](https://github.com/Alishahryar1/free-claude-code/issues)
- Yeni LLM sağlayıcıları ekleyin (Groq, Together AI vb.)
- Yeni mesajlaşma platformları ekleyin (Slack vb.)
- Test kapsamını artırın
- Şu an için Docker entegrasyon PR'ları kabul edilmiyor

```bash
git checkout -b benim-ozelligim
uv run ruff format && uv run ruff check && uv run ty check && uv run pytest
# Pull request açın
```

---

## Lisans

MIT Lisansı. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.

[FastAPI](https://fastapi.tiangolo.com/), [OpenAI Python SDK](https://github.com/openai/openai-python), [discord.py](https://github.com/Rapptz/discord.py) ve [python-telegram-bot](https://python-telegram-bot.org/) ile geliştirilmiştir.
