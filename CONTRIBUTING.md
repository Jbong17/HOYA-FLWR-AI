# Contributing to HOYA-FLWR-AI

Thank you for your interest in contributing to the Philippine Hoya Clade Classifier project! 🌺

## How to Contribute

### 1. Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in [Issues](https://github.com/Jbong17/HOYA-FLWR-AI/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)

### 2. Contributing Data

We're actively seeking more Hoya pollinarium specimens!

**What we need:**
- Microscopic measurements of pollinaria (10 measurements per specimen)
- Species identification (confirmed by expert taxonomist)
- Geographic origin information
- High-resolution images (optional but valuable)

**Data format:**
- CSV or Excel format preferred
- Follow the structure in `data/template.csv` (if available)
- Include metadata: collector, date, location, herbarium voucher

**How to contribute data:**
1. Email measurements to [your-email]
2. Or create a Pull Request adding to `data/contributions/`
3. Ensure all specimens are ethically sourced

### 3. Code Contributions

We welcome improvements to:
- Model performance (new algorithms, hyperparameter tuning)
- Feature engineering (novel morphometric features)
- Web interface (UI/UX improvements)
- Documentation (tutorials, examples, translations)
- Deployment tools (Docker, API endpoints)

**Before you start:**
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a Pull Request

**Code standards:**
- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

### 4. Areas of Interest

**High Priority:**
- [ ] Additional specimen data (especially Centrostemma and Pterostelma)
- [ ] Species-level classification models
- [ ] Mobile app development
- [ ] Integration with herbarium databases

**Medium Priority:**
- [ ] Hyperparameter optimization
- [ ] Alternative ensemble methods
- [ ] Web interface improvements
- [ ] Multi-language support

**Nice to Have:**
- [ ] Image-based classification (pollinarium photos)
- [ ] Molecular marker integration
- [ ] Geographic distribution modeling
- [ ] Automated measurement extraction from images

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/HOYA-FLWR-AI.git
cd HOYA-FLWR-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (when available)
pytest tests/

# Run web app locally
streamlit run app.py
```

## Pull Request Process

1. Update README.md with details of changes (if applicable)
2. Update requirements.txt if you add dependencies
3. Ensure all tests pass
4. Reference any related issues in your PR description
5. Request review from maintainers

## Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity, gender identity
- Level of experience, nationality, personal appearance
- Race, religion, or sexual identity and orientation

### Our Standards

**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behavior:**
- Trolling, insulting/derogatory comments, personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Academic papers citing significant contributions
- Conference presentations

## Questions?

Feel free to:
- Open an issue for questions
- Email the maintainer: [your-email]
- Start a discussion in [Discussions](https://github.com/Jbong17/HOYA-FLWR-AI/discussions)

---

**Thank you for helping preserve Philippine biodiversity through AI! 🌺**
