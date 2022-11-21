pipeline {
  agent any
    stages {
      stage('Code Quality Check via SonarQube') {
        steps {
          script {
            def scannerHome = tool 'SonarQube';
                withSonarQubeEnv('SonarQube') {
                    sh "${scannerHome}/bin/sonar-scanner -Dsonar.projectKey=3104_Sonar -Dsonar.sources=."
                }
            }
          }
        }
      }
  post {
    always {
      recordIssues enabledForFailure: true, tool: sonarQube()
    }
  }
}
