// Copyright 2020 CS GROUP - France, http://www.c-s.fr
// All rights reserved

pipeline {
    agent { label 'hpc' }
    parameters {
        string(name: 'p_gitlab_SourceBranch', defaultValue: 'Unkown', description: 'gitlabSourceBranch')
        string(name: 'p_gitlab_SourceRepoHttpUrl', defaultValue: 'Unkown', description: 'gitlabSourceRepoHttpUrl')
        string(name: 'p_gitlab_ActionType', defaultValue: 'Unkown', description: 'gitlabActionType')
    }
    options {
        gitLabConnection('gitlab.cnes.fr')
        gitlabBuilds(builds: ['checkout', 'run tests'])
        skipDefaultCheckout true
    }
    environment {
        registry_url = 'artifactory.cnes.fr'
        ARTIFACTORY_CREDS = credentials('artifactory-credentials')
        gitlabSourceBranch = "${env.gitlabSourceBranch == null ? "${p_gitlab_SourceBranch}" : "${env.gitlabSourceBranch}"}"
        gitlabSourceRepoHttpUrl = "${env.gitlabSourceRepoHttpUrl == null ? "${p_gitlab_SourceRepoHttpUrl}" : "${env.gitlabSourceRepoHttpUrl}"}"
        gitlabActionType = "${env.gitlabActionType == null ? "${p_gitlab_ActionType}" : "${env.gitlabActionType}"}"
    }
    stages{
        stage("checkout") {
            steps {
                script {
                    deleteDir()
                    sh 'mkdir -p certs'
                    sh 'cp /etc/pki/ca-trust/source/anchors/AC*.crt certs/'
                    sh 'printenv'
                    echo "${env.gitlabSourceBranch}"
                    echo "${env.gitlabSourceRepoHttpUrl}"
                     
                    checkout scm: [
                        $class: 'GitSCM',
                        clearWorkspace: true,
                        branches: [[name: "origin/${env.gitlabSourceBranch}"]],
                        doGenerateSubmoduleConfigurations: false,
                        extensions: [[$class : 'WipeWorkspace'],
                                     [$class : 'SubmoduleOption',
                                     disableSubmodules: false,
                                     parentCredentials: false,
                                     recursiveSubmodules: false,
                                     references: '',
                                     trackingSubmodules: false]],
                        userRemoteConfigs: [[name: 'origin',
                                             url: "${env.gitlabSourceRepoSshUrl}",
                                             refspec: "+refs/heads/*:refs/remotes/origin/* +refs/tags/*:refs/remotes/origin/refs/tags/*",
                                             credentialsId: "bulldozer-access-key",
                                             ]],
                    ]
                }
                script{ author_email = sh(returnStdout: true, script: 'git log -1 --pretty=format:"%ae"') }
            }
            post {
                failure {
                    updateGitlabCommitStatus name: 'checkout', state: 'failed'
                }
                success {
                    updateGitlabCommitStatus name: 'checkout', state: 'success'
                }
            }
        }
        stage('build-app') {
            steps {
                sh """
                module purge
                module load python
                make install
                """
            }
            post {
                failure {
                    updateGitlabCommitStatus name: 'build-app', state: 'failed'
                }
                success {
                    updateGitlabCommitStatus name: 'build-app', state: 'success'
                }
            }
        }
        stage('run-tests') {
            steps {
                sh """
                source .venv/bin/activate
                make test
                """
            }
            post {
                failure {
                    updateGitlabCommitStatus name: 'run-tests', state: 'failed'
                }
                success {
                    updateGitlabCommitStatus name: 'run-tests', state: 'success'
                }
            }
        }
        stage("pip-deploy") {
            when { expression { return env.gitlabActionType == "TAG_PUSH"}}
            // push pip artefact
            steps{
                sh """
                # create venv with twine to push with customed .pypirc location
                module load python 1>/tmp/null 2>&1
                virtualenv venv_twine
                source venv_twine/bin/activate
                pip install twine
                 
                # create pypirc file
                cat >.pypirc <<EOF
[distutils]
index-servers = ai4geo_cnes_artifactory
[ai4geo_cnes_artifactory]
repository: https://artifactory.cnes.fr/artifactory/api/pypi/ai4geo-pip
username: $ARTIFACTORY_CREDS_USR
password: $ARTIFACTORY_CREDS_PSW
EOF
                # change version according to the tag
                #   remove obsolete version
                sed -i "/version=/d" setup.py
                #   add version= using tag value
                cat setup.py
                sed -i 's|long_description=readme(),|long_description=readme(),\\n      version='"'`basename ${env.gitlabSourceBranch}`'"',|' setup.py
                cat setup.py
 
                # create distrib
                python setup.py sdist bdist_wheel
                 
                # push to artifactory with twine
                twine upload --config-file .pypirc -r ai4geo_cnes_artifactory dist/*
                rm .pypirc
                """
            }
            post {
                failure {
                    updateGitlabCommitStatus name: 'pip-deploy', state: 'failed'
                }
                success {
                    updateGitlabCommitStatus name: 'pip-deploy', state: 'success'
                }
            }
        }
    }
    post {
        success {
            emailext (
                subject: "[Jenkins] SUCCESSFUL: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                body: """Package successfully saved into artifactory.""",
                to: "${author_email}"
            )
        }
        failure {
            emailext (
                subject: "[Jenkins] FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
                body: """Build failed. Package not saved into artifactory.""",
                to: "${author_email}"
            )
        }
    }
}
