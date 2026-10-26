from pydantic import BaseModel, Field


class RespuestaRAG(BaseModel):
    respuesta: str = Field(
        description="Respuesta precisa basada EXCLUSIVAMENTE en el contexto proporcionado. Si la información no está en el contexto, debe indicar que no la posee."
    )
    fuentes_utilizadas: list[str] = Field(
        default=[],
        description="Lista de fragmentos o referencias extraídas del contexto utilizadas para responder.",
    )