# Development Assistance Notes

AI-assisted coding was used during development for scaffolding, debugging, and documentation cleanup. The final training runs, model metrics, app behavior, sample outputs, and repository contents were checked locally before submission.

## Prompts Used

Representative prompts used during development:

- Build a small Streamlit application for traffic-camera vehicle detection and traffic-density labeling based on the provided problem statement.
- Prepare a reproducible BMD-45 subset pipeline that converts COCO-style annotations into YOLO format.
- Add command-line inference that accepts one image or a folder and writes CSV/JSON predictions plus annotated sample images.
- Run a short adaptation training step with YOLO and document epochs, subset size, training command, and metrics.
- Compare the YOLO detector with a transformer-style detector such as RT-DETR and summarize limitations.
- Clean the repository for GitHub submission and keep datasets, virtual environments, logs, and training runs out of version control.

## Review Notes

The project was reviewed against the problem statement requirements:

- data preparation
- short adaptation training
- image and folder inference
- CSV output
- annotated sample images
- Streamlit interface
- density-label logic
- setup and run instructions
- assumptions, limitations, and next steps
