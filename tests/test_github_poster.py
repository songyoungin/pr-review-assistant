import pytest


def test_parse_github_pr_url():
    from tools.git.provider_github import parse_github_pr_url

    owner, repo, num = parse_github_pr_url("https://github.com/serena/myrepo/pull/42")
    assert owner == "serena"
    assert repo == "myrepo"
    assert num == 42


def test_post_comment_happy_path(monkeypatch):
    from tools.git.provider_github import GitHubPoster

    called = {}

    class FakeResp:
        def __init__(self):
            self.status_code = 201
            self.ok = True
            self.text = "created"

        def json(self):
            return {"id": 1, "body": "ok"}

    def fake_post(url, headers=None, json=None, timeout=None):
        called["url"] = url
        called["headers"] = headers
        called["json"] = json
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    poster = GitHubPoster(token="tok")
    res = poster.post_comment("org", "repo", 123, "hello")
    assert res["id"] == 1
    assert called["url"].endswith("/repos/org/repo/issues/123/comments")
    assert "Authorization" in called["headers"]


def test_post_comment_auth_failure(monkeypatch):
    from tools.git.provider_github import GitHubPoster

    class FakeResp:
        def __init__(self):
            self.status_code = 401
            self.ok = False
            self.text = "unauthorized"

        def json(self):
            return {"message": "unauthorized"}

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    poster = GitHubPoster(token="tok")
    with pytest.raises(RuntimeError):
        poster.post_comment("org", "repo", 1, "hi")
