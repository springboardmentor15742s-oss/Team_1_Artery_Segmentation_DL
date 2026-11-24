def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_register_login_me(client):
    r = client.post("/api/auth/register", json={"email": "A@Example.com", "full_name": "A",
                                                 "password": "password123"})
    assert r.status_code == 201 and r.json()["role"] == "researcher"
    r = client.post("/api/auth/login", data={"username": "a@example.com", "password": "password123"})
    assert r.status_code == 200
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert client.get("/api/auth/me", headers=headers).json()["email"] == "a@example.com"


def test_duplicate_email_and_bad_password(client):
    body = {"email": "a@example.com", "full_name": "A", "password": "password123"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 400
    assert client.post("/api/auth/register", json={**body, "email": "b@example.com", "password": "short"}).status_code == 422
    r = client.post("/api/auth/login", data={"username": "a@example.com", "password": "wrong-password"})
    assert r.status_code == 401


def test_public_register_cannot_create_admin(client):
    r = client.post("/api/auth/register", json={"email": "x@example.com", "full_name": "X",
                                                 "password": "password123", "role": "admin"})
    assert r.json()["role"] == "researcher"


def test_token_required(client):
    assert client.get("/api/patients").status_code == 401
    assert client.get("/api/patients", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_admin_can_change_role(client, admin, researcher):
    users = client.get("/api/users", headers=admin).json()
    target = next(u for u in users if u["email"] == "res@example.com")
    r = client.patch(f"/api/users/{target['id']}/role", json={"role": "clinician"}, headers=admin)
    assert r.status_code == 200 and r.json()["role"] == "clinician"
    assert client.get("/api/users", headers=researcher).status_code == 403
