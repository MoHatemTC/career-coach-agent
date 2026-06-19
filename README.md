# Career Coach

An AI-powered system that helps users build profiles, discover jobs, and get personalized career recommendations.

## Roadmap

### Week 1: Profile & CV
- CV upload flow
- Collect career preferences
- Generate structured user profile (JSON)

### Week 2: Job Collection Pipeline
- Connect to job sources
- Clean and normalize job data
- Remove duplicate jobs

### Week 3: AI Matching & Recommendations
- Match jobs with user profile
- Generate match scores + explanations
- Suggest missing skills

### Week 4: Application Support & Demo
- CV tailoring suggestions
- Cover letter draft (human-reviewed)
- Final demo + GitHub preparation

## Setup

### Installing uv

uv is a fast Python package installer and resolver. Follow the instructions below based on your operating system:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more information and alternative installation methods, visit the [official uv documentation](https://docs.astral.sh/uv/getting-started/installation).

### Project Setup

```bash
# Clone the repo
git clone https://github.com/atef199/career-coach.git
cd career-coach

# Install dependencies using uv
uv sync