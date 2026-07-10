# Docs Overview

This documentation describes the current GDKVM/DPFR project surface.

Recommended order:

1. [architecture.md](architecture.md): package boundaries and run artifacts.
2. [config-guide.md](config-guide.md): Hydra config layout and no-leak defaults.
3. [develop-modules.md](develop-modules.md): adding future methods through registries.
4. [develop-datasets.md](develop-datasets.md): adding datasets and protocols.
5. [logging-guide.md](logging-guide.md): MLflow/local artifact conventions.
6. [project-guidelines.md](project-guidelines.md): engineering and testing norms.

New capabilities should enter through the `gdkvm_project` public package and its
registries. Legacy top-level method packages are not public extension points.
