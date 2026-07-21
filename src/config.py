from pathlib import Path

APP_NAME = "Reto_Irrelevant"
ROOT_DIR = Path(__file__).resolve().parents[1]
DEMO_DATA_PATH = ROOT_DIR / "data" / "pedidos_mock.csv"
CSS_PATH = ROOT_DIR / "assets" / "styles.css"

REQUIRED_COLUMNS = ("fecha", "cliente", "producto", "valor")
MAX_FILES = 12
MAX_ROWS = 100_000
MAX_QUESTION_LENGTH = 500
MAX_CHAT_MESSAGES = 10
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "thinkingmachines/inkling"

COLUMN_ALIASES = {
    "fecha": {"fecha", "date", "fecha_pedido", "fecha venta", "fecha_venta", "order_date"},
    "cliente": {"cliente", "customer", "comprador", "razon_social", "razón social", "client"},
    "producto": {"producto", "product", "articulo", "artículo", "item", "descripcion", "descripción"},
    "valor": {"valor", "venta", "total", "importe", "monto", "sales", "amount", "valor_total"},
}
