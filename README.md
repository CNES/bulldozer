# bulldozer

## Setup

```sh
make git install
source .venv/bin/activate
make test  # for testing with the virtual env Python version
tox  # for testing with every configured Python versions
```

See `tox.ini` file for configuration of tested Python versions.

## Usage

### Adding some dependencies

Update `install_requires` section of `setup.py` file to add a dependency.

Then run `pip install --upgrade .`

## VSCode Python environment initialisation

* Select previously created Python virtual environment as Python interpreter: `CTRL+SHIFT+p` and type "interpreter"
* Configure Python test runner: `CTRL+SHIFT+p` then type "configure test" and choose Pytest
* Formatting with black: `CTRL+SHIFT+i` then choose "black" button into VSCode popup
* Linting with flake8: `CTRL+SHIFT+i` then choose "select linter" then select flake8

After that, Python tests tab appears and you can run tests.

## Test coverage

### What is Code Coverage?

Code coverage is the percentage of code which is covered by automated tests. Code coverage measurement simply determines which statements in a body of code have been executed through a test run, and which statements have not.

### Project Report

Coverage report can be found after launching ```make test``` into ```$PROJECT_HOME/htmlcov``` folder.

1) Open a Web Browser
2) Browse ```file:///$PROJECT_HOME/htmlcov/index.html```

## CI with Jenkins and Artifactory

A Jenkinsfile is generated to help you to add CI to push your package into Artifactory ```ai4geo-pip``` repo.

## Push a package into Artifactory with CI and Jenkins

- Go to your gitlab project page
- Create credentials (Settings > Repository > Deploy Keys : [tutorial](https://confluence.cnes.fr/pages/viewpage.action?pageId=24095332)). Use ```<project_name>-access-key``` as name to fit with Jenkins file or modify it (search for "credentialsId" variable into "checkout" stage and modify it with you key name)
- Connect to HAL with 3D session (with [Turbo VNC viewer](https://confluence.cnes.fr/display/AI4/3.+VNC+Session) for example)
- Open a web browser and browse Jenkins web ui (https://jenkins-ci.cnes.fr/ , only accessible from CNES network)
- Inside Jenkins open AI4GEO folder (https://jenkins-ci.cnes.fr/job/ai4geo/ , only accessible from CNES network)
- Depending on your project go to LOTx folder
- Create a new pipeline :
  - Click "new item" (on the left icon menu)
    - add "Project name"
    - use "Copie depuis". A Template named "TemplateJenkisCI" is located into AI4GEO folder. So if you want to create a project into "LOT5" just add "../TemplateJenkinsCI" (relative path to the template).
    - click "OK"
  - A new wizard will open :
    - Build Triggers section :
      - Pay attention to webhook url into "Build Triggers" section (looks like : ``` https://jenkins-ci.cnes.fr/project/ai4geo/lot<X>/<PipelineName> ``` , needed later).
    - Advanced Project Options:
      - Add your repository url : git@gitlab.cnes.fr:path (Example for [vreai4geo](https://gitlab.cnes.fr/ai4geo/lot4/vreai4geo) :```git@gitlab.cnes.fr:ai4geo/lot4/vreai4geo.git```)
      - Select your credentials (previously created) from credentials scroll bar.
      - Let "*/master" (that also means that your tags needed to be created from master, change branch name if you want to create a tag from an other branch).
    - Click "Save"
- Go back to Gitlab and add a webhook (Settings > Webhooks):
  - Copy the webhook address given by Jenkins into "URL"
  - Let "Secret Token" empty
  - Only check "Push events" and "Enable SSL Verification"
  - Click "Add Webhook"
- Push a tag (created from master) and let it run (push into Artifactory is managed into Jenkinsfile.gvy, you'll have nothing to do)
- An email is sended after the build to inform you of the result

Inspired by [CI with Jenkins tutorial](https://confluence.cnes.fr/display/AI4/Tutorial+-+CI+with+Jenkins).

Note: Jenkins build will use tag version name to push into Artifactory. Please make sure that this tag name is a correct python package tag.


## Credits

This package was created with Cookiecutter and the [ai4geo/cookiecutter-python](https://gitlab.cnes.fr/ai4geo/lot2/cookiecutter-python) project template.

## Contributing

Commit messages follow rules defined by [Conventional Commits](https://www.conventionalcommits.org).

*Copyright 2021 PIERRE LASSALLE  
All rights reserved*
