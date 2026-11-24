PATIENT = {"patient_code": "P-0001", "age": 61, "gender": "female", "risk_factors": "hypertension"}


def make_patient(client, headers, code="P-0001"):
    r = client.post("/api/patients", json={**PATIENT, "patient_code": code}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def make_case(client, headers, patient_id, code="C-0001"):
    r = client.post("/api/cases", json={"case_code": code, "patient_id": patient_id}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def test_patient_crud_and_roles(client, admin, researcher, scientist):
    p = make_patient(client, researcher)
    assert client.post("/api/patients", json={**PATIENT, "patient_code": "P-9"}, headers=scientist).status_code == 403
    assert client.post("/api/patients", json=PATIENT, headers=researcher).status_code == 400  # duplicate code
    assert client.get(f"/api/patients/{p['id']}", headers=scientist).status_code == 200
    r = client.put(f"/api/patients/{p['id']}", json={**PATIENT, "age": 62}, headers=researcher)
    assert r.json()["age"] == 62
    assert client.delete(f"/api/patients/{p['id']}", headers=researcher).status_code == 403
    assert client.delete(f"/api/patients/{p['id']}", headers=admin).status_code == 204
    assert client.get(f"/api/patients/{p['id']}", headers=admin).status_code == 404


def test_case_crud(client, admin, researcher):
    p = make_patient(client, researcher)
    c = make_case(client, researcher, p["id"])
    assert c["status"] == "created" and c["modality"] == "ccta" and c["images"] == []
    assert client.post("/api/cases", json={"case_code": "C-0001", "patient_id": p["id"]}, headers=researcher).status_code == 400
    assert client.post("/api/cases", json={"case_code": "C-2", "patient_id": 999}, headers=researcher).status_code == 404
    r = client.put(f"/api/cases/{c['id']}", json={"coronary_dominance": "right"}, headers=researcher)
    assert r.json()["coronary_dominance"] == "right"
    assert len(client.get("/api/cases", params={"patient_id": p["id"]}, headers=researcher).json()) == 1
    assert client.get("/api/cases", params={"status": "analyzed"}, headers=researcher).json() == []
    assert client.delete(f"/api/cases/{c['id']}", headers=admin).status_code == 204


def test_validation_errors(client, researcher):
    r = client.post("/api/patients", json={"patient_code": "", "age": 500}, headers=researcher)
    assert r.status_code == 422
    assert client.get("/api/patients", params={"limit": 1000}, headers=researcher).status_code == 422
