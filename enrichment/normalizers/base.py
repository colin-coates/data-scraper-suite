from typing import Dict, Any

class BaseNormalizer:
    def normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
