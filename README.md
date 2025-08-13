# SimBench

**Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants?**

## Welcome!

This repository contains the code and data for our paper exploring the reliability of user simulators in evaluating AI assistants through multi-turn conversations.

## Repository Structure

### 📁 `data/`
Contains real human–AI dialogues for two tasks:
- **Math tutoring**
- **Document creation**

Each conversation is fully annotated. The directory also includes GPT-4o extracted user profiles in the `user_profiles/` subfolder based on the conversations, which can be used in the user simulator to create more realistic user behaviors.

### 📁 `simulation/`
Contains the code for simulating users and conducting conversations with AI assistants. 

To run simulations:
```bash
# For math tutoring task
bash user_simulation_math_tutoring.sh

# For document creation task
bash user_simulation_document_creation.sh
```

### 📁 `crowdsourcing/`
Includes resources for human evaluation:
- Web interface code
- Heroku deployment scripts
- Amazon Mechanical Turk job launching scripts


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
