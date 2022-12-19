# **BULLDOZER** **Contributing guide**.

1. [Bug report](#bug-report)
2. [Contributing workflow](#contributing-workflow)
3. [Merge request acceptation process](#merge-request-acceptation-process)

# Bug report

Any proven or suspected malfunction should be traced in a bug report, the latter being an issue in the BULLDOZER github repository.

**Don't hesitate to do so: It is best to open a bug report and quickly resolve it than to let a problem remains in the project.**
**Notifying the potential bugs is the first way for contributing to a software.**

In the problem description, be as accurate as possible. Include:
* The procedure used to initialize the environment
* The incriminated command line or python function
* The content of the input YAML configuration file

# Contributing workflow

Any code modification requires a Merge Request. It is forbidden to push patches directly into master (this branch is protected).

It is recommended to open your Merge Request as soon as possible in order to inform the developers of your ongoing work.
Please add `WIP:` before your Merge Request title if your work is in progress: This prevents an accidental merge and informs the other developers of the unfinished state of your work.

The Merge Request shall have a short description of the proposed changes. If it is relative to an issue, you can signal it by adding `Closes xx` where xx is the reference number of the issue.

Likewise, if you work on a branch (which is recommended), prefix the branch's name by `xx-` in order to link it to the xx issue.

BULLDOZER Classical workflow is :
* Create an issue (or begin from an existing one)
* Create a Merge Request from the issue: a MR is created accordingly with "WIP:", "Closes xx" and associated "xx-name-issue" branch
* BULLDOZER hacking code from a local working directory following the bulldozer development rules
* Git add, commit and push from local working clone directory or from the forge directly
* Follow [Conventional commits](https://www.conventionalcommits.org/) specifications for commit messages
* Launch the tests on your modifications (or don't forget to add ones).
* When finished, change your Merge Request name (erase "WIP:" in title ) and ask the maintainers to review the code (see below Merge request acceptation process)


# Merge request acceptation process

The Merge Request will be merged into master after being reviewed by BULLDOZER steering committee (core committers) composed of:
* Dimitri Lallement (CNES)
* Pierre Lassalle (CNES)

Only the members of this committee can merge into master after reviewing your code.