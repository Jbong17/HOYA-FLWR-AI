# Deployment Guide

This guide explains how to deploy the Hoya Clade Classifier web application.

## Option 1: Streamlit Cloud (Recommended - Free!)

### Prerequisites
- GitHub account
- Streamlit Cloud account (sign up at https://streamlit.io/cloud)

### Steps

1. **Ensure all files are committed to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Sign in to Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub

3. **Create new app:**
   - Click "New app"
   - Repository: `Jbong17/HOYA-FLWR-AI`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

4. **App URL:**
   - Your app will be available at: `https://jbong17-hoya-flwr-ai-app-xxxxx.streamlit.app/`
   - Share this URL with users!

### Important Notes

**Model File Size:**
- Streamlit Cloud has a 1GB limit
- If `hoya_clade_classifier_production.pkl` > 100MB:
  - Host on external storage (Google Drive, Hugging Face)
  - Modify `app.py` to download model on startup:

```python
import gdown

@st.cache_resource
def load_model():
    # Download from Google Drive if not present
    if not os.path.exists('hoya_clade_classifier_production.pkl'):
        url = 'YOUR_GOOGLE_DRIVE_SHARE_LINK'
        output = 'hoya_clade_classifier_production.pkl'
        gdown.download(url, output, quiet=False)
    
    with open('hoya_clade_classifier_production.pkl', 'rb') as f:
        return pickle.load(f)
```

---

## Option 2: Local Deployment

### For Development/Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## Option 3: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY hoya_clade_classifier_production.pkl .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t hoya-classifier .

# Run container
docker run -p 8501:8501 hoya-classifier
```

Access at: `http://localhost:8501`

---

## Option 4: Heroku Deployment

### Prerequisites
- Heroku account
- Heroku CLI installed

### Additional Files Needed

**Procfile:**
```
web: sh setup.sh && streamlit run app.py
```

**setup.sh:**
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Deploy

```bash
heroku login
heroku create hoya-classifier
git push heroku main
heroku open
```

---

## Option 5: AWS EC2 Deployment

### Launch EC2 Instance
- AMI: Ubuntu 20.04
- Instance type: t2.micro (free tier)
- Security group: Allow port 8501

### Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip -y
pip3 install -r requirements.txt

# Run with screen
screen -S streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
# Detach: Ctrl+A, then D
```

Access at: `http://your-ec2-ip:8501`

---

## Option 6: Google Cloud Run

### Containerize and Deploy

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/YOUR_PROJECT/hoya-classifier

# Deploy to Cloud Run
gcloud run deploy hoya-classifier \
  --image gcr.io/YOUR_PROJECT/hoya-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Monitoring & Maintenance

### Streamlit Cloud

**View logs:**
- Click "Manage app" in Streamlit Cloud dashboard
- View logs in real-time

**Update app:**
- Simply push changes to GitHub
- Streamlit Cloud auto-deploys on new commits

### Performance Tips

1. **Use caching:**
   ```python
   @st.cache_resource
   def load_model():
       # Model loading code
   ```

2. **Optimize model size:**
   - Consider model compression
   - Use pickle protocol 4+

3. **Monitor usage:**
   - Check Streamlit Cloud analytics
   - Set up error alerts

### Custom Domain (Optional)

**Streamlit Cloud:**
- Upgrade to Pro plan
- Add custom domain in settings

**Self-hosted:**
- Use nginx as reverse proxy
- Set up SSL with Let's Encrypt

---

## Troubleshooting

### Common Issues

**"Model file not found"**
- Ensure `hoya_clade_classifier_production.pkl` is in repository
- Check file path in `app.py`

**"Module not found"**
- Verify all dependencies in `requirements.txt`
- Check Python version (3.8+)

**App crashes on startup**
- Check logs for stack trace
- Verify model file isn't corrupted

**Slow loading**
- Model file too large - consider compression
- Use `@st.cache_resource` decorator

---

## Security Considerations

1. **Never commit secrets:**
   - Use environment variables
   - Use Streamlit secrets management

2. **Rate limiting:**
   - Implement if hosting publicly
   - Prevent abuse

3. **Input validation:**
   - Already implemented in `app.py`
   - Sanitize user inputs

4. **HTTPS:**
   - Streamlit Cloud provides automatic HTTPS
   - Self-hosted: use nginx + Let's Encrypt

---

## Next Steps After Deployment

1. **Test thoroughly:**
   - Try various inputs
   - Check on different devices

2. **Gather feedback:**
   - Share with colleagues
   - Collect user suggestions

3. **Monitor performance:**
   - Track prediction accuracy
   - Log misclassifications for retraining

4. **Iterate:**
   - Update model with new data
   - Improve UI based on feedback

---

## Support

For deployment issues:
- GitHub Issues: https://github.com/Jbong17/HOYA-FLWR-AI/issues
- Email: [your-email]

---

**Ready to deploy? Start with Streamlit Cloud for the easiest setup! 🚀**
