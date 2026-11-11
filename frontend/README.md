# ShopSense AI frontend

A React + Vite + TypeScript frontend for the existing ShopSense AI FastAPI service.

## Design direction

The interface uses an editorial “signal map” motif for the empty state: the visual language is specific to multimodal product intelligence, while the rest of the UI stays restrained and practical. The palette pairs ice neutrals with ink, indigo, and a single coral signal marker. Space Grotesk provides the product voice and DM Sans keeps data legible.

The project-local design skill is installed at `.agents/skills/frontend-design/SKILL.md` from Anthropic’s `anthropics/skills` repository.

## Run locally

```bash
npm install
npm run dev
```

Copy `.env.example` to `.env.local` if the FastAPI service is not running at `http://localhost:8000`.

The frontend sends `multipart/form-data` to `POST /api/v1/analyze` with `file` and `top_k=5`. The typed adapter accepts the common classification keys (`classification`, `predicted_category`, `confidence`) and retrieval keys (`retrievals`, `retrieval_results`, `similar_products`, `results`). Product images are rendered only when the backend returns an image URL or safe image endpoint; no stock or synthetic imagery is used.

## Backend integration

The backend now returns `image_url` for retrieved IDs when the cached Fashion Product Images dataset is available. The safe image route is `GET /api/v1/products/{product_id}/image`; it never accepts a client-supplied filesystem path. The frontend keeps a quiet “No image” state if an image is unavailable.
