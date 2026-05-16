import os
import requests
import json


class ArgosApiClient:
    def __init__(self, base_url="http://localhost:8080", username="argos_device_01", password="senha_segura_ia_2026"):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token = None

    def autenticar(self):
        """Faz login na API Spring Boot e armazena o token JWT na memória."""
        url_login = f"{self.base_url}/api/auth/login"
        payload = {
            "username": self.username,
            "password": self.password
        }

        print("🔐 [Autenticação] Tentando obter Token JWT com o backend...")
        try:
            response = requests.post(url_login, json=payload, timeout=10)
            if response.status_code == 200:
                self.token = response.json().get("token")
                print("✅ [Autenticação Sucesso] Token JWT recebido e armazenado.")
                return True
            else:
                print(f"❌ [Autenticação Falhou] Status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ [Autenticação Erro] Não foi possível conectar ao servidor: {e}")
            return False

    def obter_headers(self):
        """Monta o cabeçalho HTTP injetando o Bearer Token."""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def enviar_deteccao(self, postura, client_id, video_path, info_adicional="", registrar_log_fn=None):
        """Executa o fluxo de duas etapas para envio da queda para o Spring Boot."""
        # Se não temos token, tenta re-autenticar antes de enviar
        if not self.token:
            if not self.autenticar():
                print("⚠️ [Envio Abortado] Impossível enviar dados sem autenticação válida.")
                return None

        url_detections = f"{self.base_url}/api/detections"
        payload = {
            "posture": postura,
            "client": {"id": client_id},
            "additionalInfo": info_adicional
        }

        print(f"📡 [Etapa 1] Enviando metadados do alerta para o Cliente ID {client_id}...")
        try:
            # Envia os metadados com o Token no Header
            response = requests.post(url_detections, json=payload, headers=self.obter_headers(), timeout=15)

            # Tratativa para token expirado: Tenta renovar uma vez se der 403 ou 401
            if response.status_code in [401, 403]:
                print("🔄 [JWT Expirado] Tentando re-autenticar automaticamente...")
                if self.autenticar():
                    response = requests.post(url_detections, json=payload, headers=self.obter_headers(), timeout=15)

            if response.status_code == 200:
                evento_salvo = response.json()
                evento_id = evento_salvo.get("id")
                print(f"✅ [Etapa 1 Sucesso] Evento registrado com ID [{evento_id}].")

                # --- ETAPA 2: Upload do Vídeo ---
                if os.path.exists(video_path):
                    print(f"📦 [Etapa 2] Iniciando upload do vídeo: {os.path.basename(video_path)}...")
                    url_upload_video = f"{url_detections}/{evento_id}/video"

                    with open(video_path, 'rb') as video_file:
                        files = {
                            'file': (os.path.basename(video_path), video_file, 'video/mp4')
                        }
                        response_video = requests.patch(url_upload_video, files=files, headers=self.obter_headers(),
                                                        timeout=60)

                        if response_video.status_code == 200:
                            print(f"✅ [Etapa 2 Sucesso] Vídeo sincronizado ao evento {evento_id} com êxito!")
                            if registrar_log_fn:
                                registrar_log_fn(f"Alerta completo enviado com sucesso. ID Evento: {evento_id}")
                            return response_video.status_code
                        else:
                            print(f"⚠️ Erro na Etapa 2 (Upload) [{response_video.status_code}]: {response_video.text}")
                            if registrar_log_fn:
                                registrar_log_fn(f"Erro no upload do vídeo para o Evento ID {evento_id}")
                            return response_video.status_code
                else:
                    print(f"⚠️ Alerta: Arquivo de vídeo local não encontrado: {video_path}")
            else:
                print(f"⚠️ Erro na Etapa 1 (Criação) [{response.status_code}]: {response.text}")
                return response.status_code

        except Exception as e:
            print(f"❌ Erro crítico ao transacionar com a API: {e}")
            return None