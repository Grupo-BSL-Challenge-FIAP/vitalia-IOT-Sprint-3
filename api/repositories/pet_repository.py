import os
import oracledb

class PetRepository:
    def __init__(self):
        self.user = os.getenv("ORACLE_USER", "rm565049")
        self.password = os.getenv("ORACLE_PASSWORD", "030705")
        self.dsn = os.getenv("ORACLE_DSN", "oracle.fiap.com.br:1521/ORCL")

    def _get_connection(self):
        return oracledb.connect(
            user=self.user,
            password=self.password,
            dsn=self.dsn
        )

    def obter_informacoes_pet(self, pet_id: int):
        query = """
            SELECT PET_ID, NAME, BIRTH_DATE, SPECIES, WEIGHT_KG
            FROM TB_VITALIA_PET
            WHERE PET_ID = :pet_id
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, pet_id=pet_id)
                    row = cursor.fetchone()
                    if row:
                        return {
                            "pet_id": row[0],
                            "nome": row[1],
                            "birth_date": row[2],
                            "species": row[3],
                            "weight_kg": row[4]
                        }
        except Exception as e:
            print(f"[PetRepository] Erro ao buscar informações do pet: {e}")
        return None

    def obter_historico_medicoes(self, pet_id: int):
        query = """
            SELECT MEASUREMENT_ID, PET_ID, WEIGHT_KG, ACTIVITY_LEVEL, SLEEP_HOURS, 
                   HEART_RATE_BPM, RESPIRATORY_RATE, TEMPERATURE_C, 
                   BEHAVIOR_SCORE, MEASUREMENT_DATE
            FROM TB_VITALIA_HEALTH_MEASUREMENT
            WHERE PET_ID = :pet_id
            ORDER BY MEASUREMENT_DATE DESC
        """
        historico = []
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, pet_id=pet_id)
                    columns = [col[0].lower() for col in cursor.description]
                    for row in cursor.fetchall():
                        historico.append(dict(zip(columns, row)))
        except Exception as e:
            print(f"[PetRepository] Erro ao buscar histórico de medições: {e}")
        return historico