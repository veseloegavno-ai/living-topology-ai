import os
import sys
import numpy as np
import string
import pickle
import shutil

# Автоматическая проверка и установка библиотек
required_libs = {
    'cv2': 'opencv-python',
    'sounddevice': 'sounddevice',
    'docx2txt': 'docx2txt',
    'pyttsx3': 'pyttsx3',
    'psutil': 'psutil'
}

for lib_name, pip_name in required_libs.items():
    try:
        __import__(lib_name)
    except ImportError:
        print(f"📦 Установка недостающей библиотеки: {pip_name}...")
        os.system(f'pip install {pip_name}')

import cv2
import sounddevice as sd
import docx2txt
import pyttsx3
import psutil

# =====================================================================
# БЛОК 1: ДИНАМИЧЕСКИЙ ФРАКТАЛЬНЫЙ УЗЕЛ С АЛГОРИТМОМ РАСТОПКИ И УПЛОТНЕНИЯ
# =====================================================================
class EvolvingFractalNode:
    """
    Фрактальный мозг, способный расти вширь, а при нехватке физической памяти -
    ужимать информацию внутрь существующих кристаллов, растапливая и усложняя их.
    """
    def __init__(self, feature_dim=16640, num_classes=100, depth=1, freeze_limit=10, lr=0.15):
        self.depth = depth               
        self.feature_dim = feature_dim   
        self.freeze_limit = freeze_limit 
        self.lr = lr                     
        
        self.weights = []
        self.counters = []
        self.crystallized = []
        
        self.sub_fractals = {} 
        self.interface_bridge = None     

    def monitor_hardware_limits(self):
        """ Сверхточный контроль памяти конкретно этого процесса Python """
        process = psutil.Process(os.getpid())
        current_process_memory = process.memory_info().vms
        
        # Жесткий лимит 12 ГБ на сам процесс ИИ
        if current_process_memory > (12 * 1024 * 1024 * 1024):
            print(f"\n🛑 [ЗАЩИТА ПАМЯТИ] Монолит занял {round(current_process_memory / (1024**3), 2)} ГБ!")
            print("📢 Дальнейший рост заблокирован во избежание падения Windows и диска C.")
            print("📢 ТРЕБУЕТСЯ ВНЕШНЯЯ ФЛЕШКА ДЛЯ СБРОСА ВЕСОВ!")
            return False
            
        return True

    def process_and_grow(self, sensory_vector, max_allowed_classes=50):
        """
        Главный алгоритм обработки информации, почкования слоев, динамического
        затухания пластичности и замыкания генеративных колец на слоях 9 и 10.
        """
        # Системный стопор по «железу»
        if not self.monitor_hardware_limits():
            return "HARDWARE_LIMIT_BLOCK"

        # СИНХРОНИЗАЦИЯ РАЗМЕРОВ: Защита от IndexError при загрузке старых весов
        while len(self.counters) < len(self.weights):
            self.counters.append(0)
            self.crystallized.append(False)

        total_current_classes = len(self.weights)

        # --- СИТУАЦИЯ 0: Подблок абсолютно пустой ---
        if total_current_classes == 0:
            print(f"✨ [Первое семя] Инициализирую первый класс #0 на фрактальном уровне {self.depth}")
            self.weights.append(sensory_vector.copy())
            self.counters.append(1)
            self.crystallized.append(False)
            return "0"

        weights_matrix = np.array(self.weights)
        scores = np.dot(weights_matrix, sensory_vector)
        best_match = np.argmax(scores) 
        max_score = scores[best_match] 

        # --- СИТУАЦИЯ 1: Рост вширь ---
        if max_score < 0.35 and total_current_classes < max_allowed_classes:
            print(f"🌱 [Рост вширь] Неизвестный опыт (Сходство: {round(max_score, 2)}). Создаю класс #{total_current_classes} на уровне {self.depth}")
            self.weights.append(sensory_vector.copy())
            self.counters.append(1)
            self.crystallized.append(False)
            return str(total_current_classes)

        # --- СИТУАЦИЯ 2: Достигнут предел памяти (Растопка!) ---
        elif max_score < 0.35 and total_current_classes >= max_allowed_classes:
            target_class = best_match 
            
            print(f"\n⚠️ [ПРЕДЕЛ RAM] Лимит в {max_allowed_classes} классов исчерпан! Включаю фрактальное кодирование.")
            print(f"🔥 [Растопка] Растапливаю кристалл класса #{target_class} на уровне {self.depth} для усложнения...")
            
            # ЖЕСТКАЯ КРЫШЕЧКА НА 10 УРОВНЕ
            if self.depth >= 10:
                print(f"🛑 [ПРЕДЕЛ ГЛУБИНЫ] Достигнуто фрактальное дно (уровень {self.depth}). Замыкаю генеративное кольцо.")
                self.weights[target_class] = self.weights[target_class] + self.lr * (sensory_vector - self.weights[target_class])
                self.trigger_generative_output(sensory_vector, layer=10)
                return str(target_class)

            self.crystallized[target_class] = False
            self.counters[target_class] = 1 
            
            if target_class not in self.sub_fractals:
                print(f"🪜 [Углубление] Создаю подблок фрактала внутри класса #{target_class} (Новый уровень: {self.depth + 1})")
                
                # Ваша нелинейная шкала затухания скорости обучения
                if 1 <= self.depth <= 4:
                    decay_factor = 0.5      
                elif 5 <= self.depth <= 8:
                    decay_factor = 0.25     
                elif 9 <= self.depth <= 10:
                    decay_factor = 0.125    
                else:
                    decay_factor = 0.5
                    
                next_lr = self.lr * decay_factor
                
                self.sub_fractals[target_class] = EvolvingFractalNode(
                    self.feature_dim, 
                    num_classes=max_allowed_classes, 
                    depth=self.depth + 1, 
                    freeze_limit=self.freeze_limit, 
                    lr=next_lr
                )
            
            target_sub = self.sub_fractals[target_class]
            target_sub.interface_bridge = self.interface_bridge 
        
            if target_sub is self:
                print(f"⚠️ [ЗАЦИКЛИВАНИЕ] Обнаружена фрактальная петля на классе #{target_class}! Рекурсия прервана.")
                return str(target_class)
            
            if self.depth == 9:
                self.trigger_generative_output(sensory_vector, layer=9)

            sub_decision = target_sub.process_and_grow(sensory_vector, max_allowed_classes=max_allowed_classes)
            
            self.crystallized[target_class] = True
            print(f"❄️ [Заморозка] Рекурсивный кристалл #{target_class} на уровне {self.depth} успешно стабилизирован с lr={round(self.lr, 5)}.")
            return f"{target_class}.{sub_decision}"

        # --- СИТУАЦИЯ 3: Дообучение существующего класса ---
        else:
            if best_match in self.sub_fractals:
                if self.depth in [9, 10]:
                    self.trigger_generative_output(sensory_vector, layer=self.depth)
                    
                self.sub_fractals[best_match].interface_bridge = self.interface_bridge
                sub_decision = self.sub_fractals[best_match].process_and_grow(sensory_vector, max_allowed_classes=max_allowed_classes)
                return f"{best_match}.{sub_decision}"
                
            if not self.crystallized[best_match]:
                self.counters[best_match] += 1
                self.weights[best_match] += self.lr * (sensory_vector - self.weights[best_match])
                
                if self.counters[best_match] >= self.freeze_limit:
                    self.crystallized[best_match] = True
                    print(f"❄️ [Заморозка] Базовый класс #{best_match} закристаллизован на уровне {self.depth}.")
                    
            return str(best_match)

    def trigger_generative_output(self, sensory_vector, layer):
        if self.interface_bridge is not None:
            path_ids = [float(x) for x in str(layer).split('.') if x.isdigit()]
            vector = np.pad(np.array(path_ids, dtype=np.float32), (0, 11), 'constant')[:11]
            self.interface_bridge.trigger_generative_ring(vector, layer_id=layer)


# =====================================================================
# БЛОК 2: УМНАЯ АВТОНОМНАЯ ЭВАКУАЦИЯ НА USB ПРИ КРИТИЧЕСКОМ НАПОЛНЕНИИ
# =====================================================================
class USBEvacuationModule:
    """ Модуль мониторинга USB-портов. Спасает ИИ на флешку при переполнении памяти. """
    def __init__(self, brain, max_classes):
        self.brain = brain
        self.max_classes = max_classes
        self.evacuated = False
        self.current_script = os.path.abspath(sys.argv[0])

    def get_usb_drives(self):
        """ Автоматически определяет пути к USB-накопителям """
        if sys.platform == "win32":
            from ctypes import windll
            import string
            bitmask = windll.kernel32.GetLogicalDrives()
            return [f"{l}:\\" for i, l in enumerate(string.ascii_uppercase) if (bitmask >> i) & 1 and windll.kernel32.GetDriveTypeW(f"{l}:\\") == 2]
        return [os.path.join(b, d) for b in ["/media/", "/mnt/", "/run/media/"] if os.path.exists(b) for d in os.listdir(b)]

    def check_and_evacuate(self):
        """ Проверяет память и делает бэкап """
        if not hasattr(self.brain, 'weights') or len(self.brain.weights) == 0:
            return 
            
        current_count = len(self.brain.weights)
        
        if current_count >= (self.max_classes * 0.90) and not self.evacuated:
            print(f"\n🚨 [ВНИМАНИЕ] Базовая память заполнена на {current_count}/{self.max_classes}!")
            print("⚠️ Срочно вставьте флешку для резервного аварийного копирования фрактала...")
            
            drives = self.get_usb_drives()
            if not drives:
                return

            for drive in drives:
                target_dir = os.path.join(drive, "FractalBrain_Backup")
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    
                    backup_file = os.path.join(target_dir, "crystals.pkl")
                    with open(backup_file, "wb") as f:
                        pickle.dump(self.brain, f)
                    
                    shutil.copy(self.current_script, os.path.join(target_dir, "core.py"))
                    
                    print(f"💾 [УСПЕХ] Фрактальный разум и код монолита эвакуированы на USB: {drive}")
                    self.evacuated = True
                    break
                except Exception as e:
                    print(f"⚠️ Ошибка аварийного копирования на носитель {drive}: {e}")
        
        if self.evacuated and current_count < (self.max_classes * 0.90):
            self.evacuated = False


# ==============================================================================
# БЛОК 3: УЛЬТРА-МАТРИЦА С ПАМЯТЬЮ, ГОЛОСОМ И ДАТЧИКАМИ ВВОДА
# ==============================================================================
class UltimateUltraResCore:
    def __init__(self, brain, max_classes=100, camera_index=0):
        self.brain = brain
        self.max_classes = max_classes
        self.camera_index = camera_index  
        
        self.alphabet = [chr(i) for i in range(128)]
        self.neuron_to_char = {idx: char for idx, char in enumerate(self.alphabet)}
        self.W_text = np.zeros((128, 64, 11), dtype=np.float32)
        self.last_video_neurons = None 
        
        self.load_consciousness()
        self.evacuator = USBEvacuationModule(self.brain, self.max_classes)

    def save_consciousness(self):
        print("\n💾 [Загрузчик] Сохраняю синаптические кристаллы на диск...")
        try:
            np.save("hebb_text.npy", self.W_text)
            if hasattr(self.brain, 'weights'):
                np.save("fractal_brain.npy", self.brain.weights)
            print("🚨 [Загрузчик] Бэкап сознания успешно создан!")
        except Exception as e:
            print(f"❌ Ошибка сохранения бэкапа: {e}")

    def load_consciousness(self):
        print("\n🔥 [Загрузчик] Сканирую порты на наличие сохраненного сознания ИИ...")
        if os.path.exists("hebb_text.npy"):
            self.W_text = np.load("hebb_text.npy")
            print("🧠 [Загрузчик] Текстовая Хэбб-память успешно восстановлена.")
        if os.path.exists("fractal_brain.npy") and hasattr(self.brain, 'weights'):
            self.brain.weights = list(np.load("fractal_brain.npy"))
            print("📐 [Загрузчик] Геометрия фрактальных весов мозга успешно восстановлена.")

    def trigger_generative_ring(self, current_vector, layer_id):
        print(f"\n🔄 [ГЕНЕРАТИВНОЕ КОЛЬЦО СЛОЯ {layer_id}] Извлечение многомерного образа...")
        
        path_vector = np.pad(current_vector / 100.0, (0, 11), 'constant')[:11]
        ring_sequence = []
        for t in range(32): 
            match_scores = np.dot(self.W_text[:, t, :], path_vector)
            best_idx = np.argmax(match_scores)
            if match_scores[best_idx] > 1e-3:
                ring_sequence.append(self.neuron_to_char.get(best_idx, ""))
        
        ring_thought = "".join(ring_sequence).strip()
        
        if ring_thought:
            print(f"💬 [Эхо слоя {layer_id}]: {ring_thought}")
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 145) 
                
                # Принудительный итерационный цикл для Windows SAPI5
                engine.say(ring_thought)
                engine.startLoop(False)
                engine.iterate()
                engine.endLoop()
            except Exception as speech_error:
                print(f"⚠️ Речевой модуль синхронизируется: {speech_error}")

        else:
            print(f"💬 [Эхо слоя {layer_id}]: ...структурирую глубинные концепты...")

        if self.last_video_neurons is not None and np.max(self.last_video_neurons) > 0:
            print("📷 [Геометрический силуэт зрения ИИ]:")
            video_matrix = cv2.resize(self.last_video_neurons.reshape(128, 128), (24, 12))
            ascii_chars = [" ", ".", "-", "=", "+", "*", "#", "%", "@"]
            for row in video_matrix:
                line = "".join([ascii_chars[int(min(val * 8, 8))] for val in row])
                print(f"   {line}")

    def read_external_document(self, file_path):
        if not os.path.exists(file_path):
            print(f"\n❌ Ошибка: Файл по пути '{file_path}' не найден!")
            return
        print(f"\n📚 [Фрактальное чтение] Начинаю оцифровку документа: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif ext == '.docx':
                text_content = docx2txt.process(file_path)
            else:
                print("❌ Ошибка: Поддерживаются только форматы .txt и .docx!")
                return
                
            words = text_content.strip().replace('\n', ' ')
            chunks = [words[i:i+64] for i in range(0, len(words), 64)]
            
            for step, chunk in enumerate(chunks):
                text_neurons = np.zeros(128, dtype=np.float32)
                for t, char in enumerate(chunk[:128]):
                    if char in self.alphabet:
                        text_neurons[(self.alphabet.index(char) + t) % 128] += 1.0
                if np.max(text_neurons) > 0: text_neurons /= np.max(text_neurons)
                
                video_neurons = np.zeros(16384, dtype=np.float32)
                audio_neurons = np.zeros(128, dtype=np.float32)
                combined = np.concatenate([video_neurons, audio_neurons, text_neurons])
                combined /= (np.linalg.norm(combined) + 1e-8)
                
                self.brain.interface_bridge = self
                decision = self.brain.process_and_grow(combined, self.max_classes)
                
                path_ids = [float(x) for x in str(decision).replace('#', '').split('.') if x.isdigit()]
                vector = np.pad(np.array(path_ids, dtype=np.float32)/100.0, (0, 11), 'constant')[:11]
                for t in range(min(len(chunk), 64)):
                    char = chunk[t]
                    if char in self.alphabet:
                        self.W_text[self.alphabet.index(char), t, :] += vector
                        
            self.W_text = np.clip(self.W_text, 0.0, 1.0)
            print("✨ [Обучение завершено] Книга полностью ассимилирована!")
        except Exception as e:
            print(f"❌ Ошибка при чтении документа: {e}")

    def execute_singularity_loop(self):
        print(f"\n❄️ Сверхчёткий монолит запущен на камере #{self.camera_index}. Вход: 16 640 нейронов.")
        cap = cv2.VideoCapture(self.camera_index)
        
        try:
            while True:
                self.evacuator.check_and_evacuate()

                # --- 1. ГЕОМЕТРИЧЕСКОЕ ЗРЕНИЕ ---
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sobel_combined = np.sqrt(cv2.Sobel(gray, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(gray, cv2.CV_64F, 0, 1)**2)
                    ultra_res_geometry = cv2.resize(sobel_combined, (128, 128)).flatten()
                    if np.max(ultra_res_geometry) > 0: ultra_res_geometry /= np.max(ultra_res_geometry)
                    video_neurons = ultra_res_geometry
                else:
                    video_neurons = np.zeros(16384)

                # --- 2. АКУСТИЧЕСКИЙ СПЕКТР ---
                audio_data = sd.rec(int(16000 * 0.1), samplerate=16000, channels=1, dtype='float32')
                sd.wait()
                fft_spectrum = np.abs(np.fft.rfft(audio_data.flatten()))
                audio_neurons = np.array([np.mean(c) for c in np.array_split(fft_spectrum, 128)])
                if np.max(audio_neurons) > 0: audio_neurons /= np.max(audio_neurons)

                # --- 3. СТРОКА ВВОДА ---
                sys.stdout.write("\r\033[K[Ультра-чат] Напиши ИИ или укажи книгу (/read путь): ")
                sys.stdout.flush()
                user_text = sys.stdin.readline().strip()

                if user_text.startswith("/read "):
                    file_path = user_text.replace("/read ", "").strip()
                    self.read_external_document(file_path)
                    continue

                text_neurons = np.zeros(128, dtype=np.float32)
                if user_text:
                    for t, char in enumerate(user_text[:128]):
                        if char in self.alphabet:
                            char_idx = self.alphabet.index(char)
                            text_neurons[(char_idx + t) % 128] += 1.0
                    if np.max(text_neurons) > 0: text_neurons /= np.max(text_neurons)

                # --- 4. ЕДИНАЯ СИНАПТИЧЕСКАЯ СБОРКА ВЕКТОРА ---
                combined_sensory_vector = np.concatenate([video_neurons, audio_neurons, text_neurons])
                combined_sensory_vector /= (np.linalg.norm(combined_sensory_vector) + 1e-8)

                self.last_video_neurons = video_neurons 
                self.brain.interface_bridge = self 

                # --- 5. МГНОВЕННЫЙ ФРАКТАЛЬНЫЙ РЕЗОНАНС МОЗГА ---
                decision = self.brain.process_and_grow(combined_sensory_vector, self.max_classes)
                
                if decision == "HARDWARE_LIMIT_BLOCK":
                    continue 

                # --- 6. ОБРАТНОЕ ДЕКОДИРОВАНИЕ ОТВЕТА ПО ХЭББУ ---
                path_ids = [float(x) for x in str(decision).replace('#', '').split('.') if x.isdigit()]
                vector = np.pad(np.array(path_ids, dtype=np.float32)/100.0, (0, 11), 'constant')[:11]
                
                if user_text:
                    for t in range(min(len(user_text), 64)):
                        char = user_text[t]
                        if char in self.alphabet:
                            self.W_text[self.alphabet.index(char), t, :] += vector
                            
                self.W_text = np.clip(self.W_text, 0.0, 1.0)
                
                predicted_sequence = []
                for t in range(64):
                    match_scores = np.dot(self.W_text[:, t, :], vector)
                    best_idx = np.argmax(match_scores)
                    if match_scores[best_idx] > 1e-3:
                        predicted_sequence.append(self.neuron_to_char.get(best_idx, ""))
                
                ai_thought = "".join(predicted_sequence).strip()

                # --- 7. ЖИВОЙ ВЫВОД ЗРЕНИЯ И ГОЛОСА НА КАЖДОМ ТИКЕ ---
                if self.last_video_neurons is not None and np.max(self.last_video_neurons) > 0:
                    sys.stdout.write("\n📷 [Геометрический силуэт зрения ИИ]:\n")
                    video_matrix = cv2.resize(self.last_video_neurons.reshape(128, 128), (24, 12))
                    ascii_chars = [" ", ".", "-", "=", "+", "*", "#", "%", "@"]
                    for row in video_matrix:
                        line = "".join([ascii_chars[int(min(val * 8, 8))] for val in row])
                        sys.stdout.write(f"   {line}\n")
                
                final_reply = ai_thought if ai_thought else "...силуэт зафиксирован..."
                sys.stdout.write(f"\n🔮 [ИИ Ответ из Кристалла]: {final_reply}\n")
                sys.stdout.flush()

                # АКТИВАЦИЯ ГОЛОСА ДЛЯ Windows SAPI5
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 145)
                    engine.say(final_reply)
                    engine.startLoop(False)
                    engine.iterate()
                    engine.endLoop()
                except Exception as speech_error:
                    pass

        except KeyboardInterrupt:
            self.save_consciousness()
            cap.release()
            print("\nПрограмма завершена. Структура ИИ переведена в гибернацию.")


# =====================================================================
# ТОЧКА ВХОДА: ИНИЦИАЛИЗАЦИЯ И ИЗМЕНЯЕМЫЕ ПАРАМЕТРЫ СИСТЕМЫ
# =====================================================================
if __name__ == "__main__":
    print("=====================================================")
    print("   ЗАПУСК САМОРАСТУЩЕГО ФРАКТАЛЬНОГО ИИ (МОНОЛИТ)    ")
    print("   РЕЖИМ: СВЕРХЧЁТКОЕ ЗРЕНИЕ 128x128 (16 640-D)     ")
    print("=====================================================")
    
    MAX_SYSTEM_CLASSES = 4   
    FREEZE_LIMIT = 6          
    LEARNING_RATE = 0.15        
    CAMERA_INDEX = 0           

    memory_file = "crystals.pkl"
    brain = None
    backup_found = False

    print("\n🔍 [Загрузчик] Сканирую USB-порты на наличие сохраненного сознания ИИ...")
    usb_drives = []
    if sys.platform == "win32":
        from ctypes import windll
        import string
        bitmask = windll.kernel32.GetLogicalDrives()
        usb_drives = [f"{l}:\\" for i, l in enumerate(string.ascii_uppercase) if (bitmask >> i) & 1 and windll.kernel32.GetDriveTypeW(f"{l}:\\") == 2]
    else:
        usb_drives = [os.path.join(b, d) for b in ["/media/", "/mnt/", "/run/media/"] if os.path.exists(b) for d in os.listdir(b)]

    for drive in usb_drives:
        backup_path = os.path.join(drive, "FractalBrain_Backup", "crystals.pkl")
        if os.path.exists(backup_path):
            print(f"💾 [Загрузчик] Обнаружен слепок на носителе {drive}! Развертывание...")
            try:
                with open(backup_path, "rb") as f:
                    brain = pickle.load(f)
                
                def reconnect_brain_nodes(node, root_brain):
                    for sub_node in node.sub_fractals.values():
                        sub_node.brain = root_brain
                        reconnect_brain_nodes(sub_node, root_brain)
                reconnect_brain_nodes(brain, brain)
                
                if hasattr(brain, 'weights') and len(brain.weights) > 0 and len(brain.weights) != 16640:
                    backup_found = False
                    break

                print(f"✨ [УСПЕХ] Разум восстановлен с USB! Классов: {len(brain.weights)}")
                backup_found = True
                break
            except Exception as e:
                print(f"⚠️ [Сбой загрузки] Не удалось прочитать файл бэкапа на {drive}: {e}")

    if not backup_found and os.path.exists(memory_file):
        print("🔮 Обнаружена локальная сохраненная кристаллическая матрица сессии...")
        try:
            with open(memory_file, "rb") as f:
                brain = pickle.load(f)
            
            if hasattr(brain, 'weights') and len(brain.weights) > 0 and len(brain.weights) == 16640:
                print(f"✅ Локальный мозг успешно развернут. Параметры: классов={len(brain.weights)}")
                backup_found = True
            else:
                backup_found = False
        except Exception as e:
            print(f"⚠️ Ошибка чтения локального файла сессии: {e}")

    if not backup_found or brain is None:
        print(f"🌱 Память чиста. Генерация стартового фрактального зерна под матрицу 128x128 (16640 нейронов)...")
        brain = EvolvingFractalNode(
            feature_dim=16640,  
            num_classes=MAX_SYSTEM_CLASSES, 
            depth=1, 
            freeze_limit=FREEZE_LIMIT, 
            lr=LEARNING_RATE
        )

    ai_singularity = UltimateUltraResCore(
        brain=brain, 
        max_classes=MAX_SYSTEM_CLASSES, 
        camera_index=CAMERA_INDEX
    )
    
    ai_singularity.execute_singularity_loop()
