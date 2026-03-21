from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Response schema for a successful file upload."""

    filename: str
    rows: int
    columns: int
    message: str
