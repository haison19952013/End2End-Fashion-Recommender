pipeline {
    agent any
    environment {
        COMPOSE_FILE = 'api-web-docker-compose.yaml'
        registry_api = 'haison19952013/fashion_recsys'
        registry_web = 'haison19952013/fashion_recsys_web'
        registryCredential = 'dockerhub'
    }
    stages {
        stage('Deploy') {
            when {
                changeset "src/**/*"  // Only run if changes are in the src folder
            }
            steps {
                script {
                    currentBuild.displayName = "Deploying Application"
                    env.CHANGE_DETECTED = 'true' // Set a flag to indicate changes were detected
                }
                sh 'docker compose -f $COMPOSE_FILE up -d'
            }
        }
    }
    post {
        success {
            script {
                if (env.CHANGE_DETECTED == 'true') {
                    echo 'Deployment completed successfully! Changes in src/ folder were detected.'
                } else {
                    echo 'No need for Deployment! No changes in src/ folder detected.'
                }
            }
        }
        failure {
            script {
                if (env.CHANGE_DETECTED == 'true') {
                    echo 'Deployment failed! Changes in src/ folder were detected.'
                } else {
                    echo 'Some errors occur! No changes in src/ folder detected.'
                }
            }
        }
        always {
            sh 'docker system prune -f'
        }
    }
}
