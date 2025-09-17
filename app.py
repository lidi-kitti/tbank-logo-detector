from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import logging
import os
import yaml
import google.generativeai as genai

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

class GeminiAnalysisResponse(BaseModel):
    """Ответ с анализом изображения от Gemini"""
    analysis: str = Field(..., description="Анализ изображения от Gemini")
    has_logo: bool = Field(..., description="Найден ли логотип Т-Банка")
    confidence: float = Field(..., description="Уверенность в наличии логотипа")
    description: str = Field(..., description="Описание найденного логотипа")
    logo_position: Optional[str] = Field(None, description="Описание расположения логотипа")
    logo_characteristics: Optional[str] = Field(None, description="Характеристики логотипа")

# Инициализация FastAPI приложения
app = FastAPI(
    title="TBank Logo Detection API",
    description="API для детекции логотипа Т-Банка на изображениях",
    version="1.0.0"
)

# Глобальные переменные
gemini_model = None
config = None

# Поддерживаемые форматы файлов
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

def load_config():
    """Загружает конфигурацию из файла"""
    global config
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Конфигурация загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        config = {}

def initialize_gemini():
    """Инициализирует Gemini API"""
    global gemini_model
    try:
        if config and 'gemini' in config:
            api_key = config['gemini']['api_key']
            model_name = config['gemini'].get('model_name', 'gemini-2.0-flash-exp')
            
            # Проверяем, является ли api_key переменной окружения
            if api_key == 'GEMINI_API_KEY':
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    logger.error("Переменная окружения GEMINI_API_KEY не установлена")
                    return
            
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini API инициализирован с моделью {model_name}")
        else:
            logger.warning("Конфигурация Gemini не найдена")
    except Exception as e:
        logger.error(f"Ошибка инициализации Gemini API: {e}")

def extract_coordinates_from_analysis(analysis_text: str, image_width: int, image_height: int) -> List[Detection]:
    """Извлекает координаты логотипов из анализа Gemini"""
    detections = []
    
    try:
        # Простой алгоритм для извлечения координат из текстового описания
        # Ищем упоминания позиций в тексте
        text_lower = analysis_text.lower()
        
        # Если логотип найден, создаем примерные координаты
        if any(keyword in text_lower for keyword in [
            "да", "yes", "логотип т-банка", "т-банк", "tbank", "найден логотип", "найден"
        ]):
            # Пытаемся извлечь более точные координаты из описания
            x_min, y_min, x_max, y_max = 0, 0, 0, 0
            
            # Простая эвристика для определения позиции
            if "лев" in text_lower or "left" in text_lower:
                x_min = 0
                x_max = image_width // 3
            elif "прав" in text_lower or "right" in text_lower:
                x_min = 2 * image_width // 3
                x_max = image_width
            else:
                x_min = image_width // 4
                x_max = 3 * image_width // 4
            
            if "верх" in text_lower or "top" in text_lower:
                y_min = 0
                y_max = image_height // 3
            elif "низ" in text_lower or "bottom" in text_lower:
                y_min = 2 * image_height // 3
                y_max = image_height
            else:
                y_min = image_height // 4
                y_max = 3 * image_height // 4
            
            # Создаем bounding box
            bbox = BoundingBox(
                x_min=max(0, x_min),
                y_min=max(0, y_min),
                x_max=min(image_width, x_max),
                y_max=min(image_height, y_max)
            )
            
            detection = Detection(bbox=bbox)
            detections.append(detection)
    
    except Exception as e:
        logger.error(f"Ошибка извлечения координат: {e}")
    
    return detections

def analyze_image_with_gemini(image_array: np.ndarray) -> GeminiAnalysisResponse:
    """Анализирует изображение с помощью Gemini API"""
    try:
        if gemini_model is None:
            raise Exception("Gemini модель не инициализирована")
        
        # Конвертируем изображение в байты
        image_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Создаем детальный промпт для анализа
        prompt = """
        Проанализируй это изображение и определи, есть ли на нем логотип Т-Банка.
        
        Т-Банк - это российский банк, их логотип обычно представляет собой:
        - Щит или прямоугольник с закругленными углами
        - Букву "Т" внутри (часто с засечками)
        - Цвета: желтый/золотой фон, белый щит с темной буквой "Т", или наоборот
        - Может быть в различных размерах и позициях
        
        ВАЖНО: Игнорируй логотипы "Тинькофф" - они НЕ являются логотипами Т-Банка!
        
        Пожалуйста, ответь в следующем структурированном формате:
        
        1. НАЙДЕН ЛИ ЛОГОТИП: [да/нет]
        2. УВЕРЕННОСТЬ: [число от 0.0 до 1.0]
        3. ОПИСАНИЕ ИЗОБРАЖЕНИЯ: [краткое описание того, что видишь на изображении]
        4. РАСПОЛОЖЕНИЕ ЛОГОТИПА: [если найден - укажи ТОЧНОЕ расположение: верхний левый, верхний правый, нижний левый, нижний правый, центр, или опиши более детально]
        5. ХАРАКТЕРИСТИКИ ЛОГОТИПА: [если найден - опиши размер, цвет, форму]
        6. ДОПОЛНИТЕЛЬНЫЕ ЗАМЕЧАНИЯ: [любая дополнительная информация]
        """
        
        # Отправляем запрос к Gemini
        response = gemini_model.generate_content([
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
        ])
        
        # Парсим ответ
        analysis_text = response.text
        
        # Улучшенный парсинг ответа
        text_lower = analysis_text.lower()
        has_logo = any(keyword in text_lower for keyword in [
            "да", "yes", "логотип т-банка", "т-банк", "tbank", "найден логотип", "найден"
        ])
        
        # Извлекаем уверенность из текста
        confidence = 0.5  # По умолчанию
        try:
            import re
            # Ищем числа от 0.0 до 1.0 в тексте
            confidence_match = re.search(r'уверенность[:\s]*(\d+\.?\d*)', text_lower)
            if confidence_match:
                conf_value = float(confidence_match.group(1))
                if conf_value <= 1.0:
                    confidence = conf_value
                elif conf_value <= 100:
                    confidence = conf_value / 100.0
        except:
            confidence = 0.8 if has_logo else 0.2
        
        # Извлекаем дополнительные поля
        logo_position = None
        logo_characteristics = None
        
        try:
            # Ищем расположение логотипа
            position_match = re.search(r'расположение[:\s]*(.+?)(?:\n|характеристики|дополнительные)', text_lower, re.DOTALL)
            if position_match:
                logo_position = position_match.group(1).strip()
            
            # Ищем характеристики логотипа
            char_match = re.search(r'характеристики[:\s]*(.+?)(?:\n|дополнительные|$)', text_lower, re.DOTALL)
            if char_match:
                logo_characteristics = char_match.group(1).strip()
        except:
            pass
        
        return GeminiAnalysisResponse(
            analysis=analysis_text,
            has_logo=has_logo,
            confidence=confidence,
            description=analysis_text,
            logo_position=logo_position,
            logo_characteristics=logo_characteristics
        )
        
    except Exception as e:
        logger.error(f"Ошибка анализа с Gemini: {e}")
        return GeminiAnalysisResponse(
            analysis=f"Ошибка анализа: {str(e)}",
            has_logo=False,
            confidence=0.0,
            description="Не удалось проанализировать изображение",
            logo_position=None,
            logo_characteristics=None
        )

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Предобработка изображения"""
    try:
        # Читаем изображение из байтов
        image = Image.open(io.BytesIO(image_bytes))
        
        # Конвертируем в RGB если необходимо
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Конвертируем в numpy array
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        logger.error(f"Ошибка предобработки изображения: {e}")
        raise HTTPException(status_code=400, detail=f"Не удалось обработать изображение: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    load_config()
    initialize_gemini()

@app.post("/detect", response_model=DetectionResponse)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    try:
        # Проверяем формат файла
        if file.content_type not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Поддерживаемые форматы: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Читаем содержимое файла
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")
        
        # Предобработка изображения
        image_array = preprocess_image(image_bytes)
        image_height, image_width = image_array.shape[:2]
        
        # Анализ с помощью Gemini
        if gemini_model is None:
            raise HTTPException(status_code=500, detail="Gemini API не инициализирован")
        
        analysis = analyze_image_with_gemini(image_array)
        
        # Извлекаем координаты из анализа
        detections = extract_coordinates_from_analysis(analysis.analysis, image_width, image_height)
        
        logger.info(f"Детекция завершена: найдено {len(detections)} логотипов, уверенность: {analysis.confidence}")
        
        return DetectionResponse(detections=detections)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при детекции логотипа: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/analyze", response_model=GeminiAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Анализ изображения с помощью Google Gemini API для поиска логотипа Т-Банка

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        GeminiAnalysisResponse: Результат анализа от Gemini AI
    """
    try:
        # Проверяем формат файла
        if file.content_type not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Поддерживаемые форматы: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Читаем содержимое файла
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")
        
        # Предобработка изображения
        image_array = preprocess_image(image_bytes)
        
        # Анализ с помощью Gemini
        if gemini_model is None:
            raise HTTPException(status_code=500, detail="Gemini API не инициализирован")
        
        analysis = analyze_image_with_gemini(image_array)
        
        logger.info(f"Gemini анализ завершен: найдено логотипов: {analysis.has_logo}, уверенность: {analysis.confidence}")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при анализе с Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy", 
        "gemini_loaded": gemini_model is not None,
        "config_loaded": config is not None
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, detail=None).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)