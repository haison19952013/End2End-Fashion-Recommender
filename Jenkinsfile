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
            steps {
                sh 'docker compose -f $COMPOSE_FILE up -d'
            }
        }
    }
    post {
        always {
            sh 'docker system prune -f'
        }
    }
}
