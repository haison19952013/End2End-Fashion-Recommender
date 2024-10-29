pipeline {
    agent any
    environment {
        COMPOSE_FILE = 'api-web-docker-compose.yaml'
        registry_api = 'haison19952013/fashion_recsys'
        registry_web = 'haison19952013/fashion_recsys_web'
        registryCredential = 'dockerhub'
    }
    stages {
        stage('Pull Latest Images') {
            steps {
                script {
                    echo "Pulling the latest images..."
                    sh "docker pull ${registry_api}:latest"
                    sh "docker pull ${registry_web}:latest"
                }
            }
        }
        stage('Deploy') {
            when {
                anyOf {
                    changeset "src/**/*"       // Trigger if changes are in the src folder
                    changeset "Jenkinsfile"     // Trigger if Jenkinsfile has changed
                }
            }
            steps {
                script {
                    currentBuild.displayName = "Deploying Application"
                    env.CHANGE_DETECTED = 'true' // Set a flag to indicate changes were detected
                }
                sh "docker compose -f ${COMPOSE_FILE} up -d"
            }
        }
    }
    post {
        success {
            script {
                if (env.CHANGE_DETECTED == 'true') {
                    echo 'Deployment completed successfully! Changes in src/ folder or Jenkinsfile were detected.'
                } else {
                    echo 'No need for Deployment! No changes in src/ folder or Jenkinsfile detected.'
                }
            }
        }
        failure {
            script {
                if (env.CHANGE_DETECTED == 'true') {
                    echo 'Deployment failed! Changes in src/ folder or Jenkinsfile were detected.'
                } else {
                    echo 'Some errors occurred! No changes in src/ folder or Jenkinsfile detected.'
                }
            }
        }
        always {
            sh 'docker system prune -f'
        }
    }
}
