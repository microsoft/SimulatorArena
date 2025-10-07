# SimulatorArena Data

This directory contains the annotated human–AI dialogue datasets for SimulatorArena evaluation framework.

## 📁 Dataset Overview

### Document Creation Task
- **File**: `document_creation_annotations.json`
- **Content**: 459 fully annotated conversations for creating blog posts, emails, and creative writing
- **Status**: ✅ Complete and ready to use

### Math Tutoring Task
- **Redacted Files**:
  - `math_tutoring_annotations_redacted.json` (450 conversations - full dataset)
  - `math_tutoring_annotations_redacted_for_benchmarking.json` (50 conversations - benchmarking subset)
- **Full Files**: `math_tutoring_annotations.json` and `math_tutoring_annotations_for_benchmarking.json` (must be generated)
- **Content**: Conversations with math problems of varying difficulty
- **Status**: ⚠️ Requires MATH dataset to restore full content

## 🔒 Important: MATH Dataset Copyright Notice

Due to copyright restrictions, we cannot directly distribute the MATH dataset problems and solutions. The redacted annotation files contain all conversation data **except** the actual math problems and solutions, which have been replaced with references to their locations in the MATH dataset.

## 🚀 How to Restore Math Tutoring Data

If you have legitimate access to the MATH dataset, you can restore the full math tutoring annotations:

### Step 1: Obtain the MATH Dataset

The MATH dataset must be obtained separately through official channels. Once you have it, place it in this directory:

```bash
SimulatorArena/data/
├── MATH/                              # Place the MATH dataset here
│   ├── train/
│   │   ├── algebra/
│   │   ├── counting_and_probability/
│   │   ├── geometry/
│   │   ├── intermediate_algebra/
│   │   ├── number_theory/
│   │   ├── prealgebra/
│   │   └── precalculus/
│   └── test/
│       └── [same structure as train]
├── math_tutoring_annotations_redacted.json
└── load_math_data.py
```

### Step 2: Run the Restoration Script

Once the MATH dataset is in place, you have two options:

**Option A: Restore both files (recommended)**
```bash
python load_math_data.py --all
```

This will restore:
- `math_tutoring_annotations_redacted.json` → `math_tutoring_annotations.json` (450 conversations)
- `math_tutoring_annotations_redacted_for_benchmarking.json` → `math_tutoring_annotations_for_benchmarking.json` (50 conversations)

**Option B: Restore a single file**
```bash
python load_math_data.py  # Restores main file only
```

This will:
1. Read `math_tutoring_annotations_redacted.json`
2. Load the referenced problems and solutions from the MATH dataset
3. Generate the complete `math_tutoring_annotations.json` file

### Alternative Usage

You can also specify custom paths:

```bash
# Restore both files with custom MATH location
python load_math_data.py --all /path/to/MATH

# Restore specific file with custom MATH location
python load_math_data.py math_tutoring_annotations_redacted.json /path/to/MATH

# Specify custom output file
python load_math_data.py -o custom_output.json

# See all options
python load_math_data.py --help
```

## 📊 Data Format

### Document Creation Annotations

Each entry contains:
- User queries and AI responses for the full conversation
- Document type (blog post, email, creative writing)
- User intent and background
- Quality ratings for both document and interaction
- Document History

### Math Tutoring Annotations

Each entry contains:
- Math problem and solution (after restoration)
- Similar follow-up problem
- Student queries and tutor responses
- Correctness indicators
- Interaction quality ratings

## 🔗 Additional Resources

### User Simulator Profiles

The `user_simulator_profiles/` subdirectory contains GPT-4o extracted user simulator profiles based on the conversations:

- **Math Tutoring Profiles**:
  - `interaction_style.json` - User interaction patterns
  - `knowledge_state.json` - Mathematical understanding levels
  - `writing_style.json` - Communication styles

- **Document Creation Profiles**:
  - `interaction_style.json` - Interaction patterns
  - `writing_style.json` - Writing characteristics  
  - `preferences.json` - Document preferences

These profiles enable more realistic user simulation by capturing individual user characteristics from the human data.

## 📝 Note on Data Usage

- The document creation dataset is immediately usable
- The math tutoring dataset requires the MATH dataset for full functionality
- All conversations are annotated with quality ratings for evaluation
- User simulator profiles are pre-extracted and ready for simulator development

For questions about data access or usage, please refer to the main repository documentation.