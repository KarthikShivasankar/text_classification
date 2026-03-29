#!/usr/bin/env python
"""Fetch issues from any public GitHub repository into a CSV file.

By default fetches the latest 100 issues. Use --all or --limit N to change this.

Usage:
    # Latest 100 issues (default)
    python scripts/fetch_github_issues.py --repo owner/repo --output issues.csv

    # Latest 500 issues
    python scripts/fetch_github_issues.py --repo owner/repo --limit 500

    # Every issue (can be slow for large repos)
    python scripts/fetch_github_issues.py --repo owner/repo --all

    # With a GitHub token (required once the unauthenticated 60 req/h limit is hit)
    python scripts/fetch_github_issues.py --repo owner/repo --token $GITHUB_TOKEN
"""

import csv
import os
import sys
import time

import requests
from tqdm import tqdm

_PAGE_SIZE = 100  # GitHub API maximum per-page


def _make_headers(token):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _handle_response_error(response, repo):
    """Raise a human-readable error for common API failures."""
    status = response.status_code
    if status == 401:
        print(
            "Error: GitHub token is invalid or expired.\n"
            "Generate a new token at https://github.com/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(1)
    if status == 403:
        # Could be rate-limited or a private repo with no token
        remaining = response.headers.get("X-RateLimit-Remaining", "?")
        reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
        wait_min = max(0, reset_ts - time.time()) / 60
        if remaining == "0" or remaining == 0:
            print(
                f"Error: GitHub API rate limit exceeded.\n"
                f"  • Unauthenticated limit: 60 requests/hour\n"
                f"  • Authenticated limit:   5000 requests/hour\n"
                f"  • Limit resets in approximately {wait_min:.0f} minute(s).\n"
                f"\nFix: supply a personal access token:\n"
                f"  python scripts/fetch_github_issues.py --repo {repo} --token $GITHUB_TOKEN\n"
                f"Create one at https://github.com/settings/tokens",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: Access forbidden (HTTP 403). The repository may be private.\n"
                f"If it is private, supply --token with a token that has repo access.",
                file=sys.stderr,
            )
        sys.exit(1)
    if status == 404:
        print(
            f"Error: Repository '{repo}' not found or is private.\n"
            f"Check the owner/repo spelling. For private repos supply --token.",
            file=sys.stderr,
        )
        sys.exit(1)
    response.raise_for_status()


def _check_rate_limit(response, repo):
    """Block if we are about to run out of API calls, with a clear message."""
    remaining = int(response.headers.get("X-RateLimit-Remaining", 999))
    if remaining < 5:
        reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(0, reset_ts - time.time()) + 2
        token_hint = "" if response.request.headers.get("Authorization") else (
            f"\nTip: use --token to raise the limit to 5000 req/h. "
            f"Create one at https://github.com/settings/tokens"
        )
        tqdm.write(
            f"Rate limit almost exhausted ({remaining} calls left). "
            f"Waiting {wait:.0f}s until reset...{token_hint}"
        )
        time.sleep(wait)


def fetch_issues(repo, state, token, limit, fetch_all):
    """Fetch issues from a GitHub repo.

    Args:
        repo: 'owner/repo' string.
        state: 'open', 'closed', or 'all'.
        token: GitHub personal access token (or None).
        limit: Maximum number of issues to return (ignored when fetch_all=True).
        fetch_all: If True, paginate through every page regardless of limit.

    Returns:
        List of issue dicts.
    """
    headers = _make_headers(token)
    base_url = f"https://api.github.com/repos/{repo}/issues"
    issues = []
    page = 1

    # GitHub always sorts newest-first by default (sort=created, direction=desc)
    with tqdm(desc="Fetching issues", unit="issue") as pbar:
        while True:
            # How many to request this page
            if fetch_all:
                page_size = _PAGE_SIZE
            else:
                remaining_needed = limit - len(issues)
                if remaining_needed <= 0:
                    break
                page_size = min(_PAGE_SIZE, remaining_needed)

            params = {
                "state": state,
                "per_page": page_size,
                "page": page,
                "sort": "created",
                "direction": "desc",
            }

            response = requests.get(base_url, headers=headers, params=params, timeout=30)
            if not response.ok:
                _handle_response_error(response, repo)

            page_data = response.json()
            if not page_data:
                break

            # The issues endpoint returns PRs too — keep only real issues
            new_issues = [item for item in page_data if "pull_request" not in item]
            issues.extend(new_issues)
            pbar.update(len(new_issues))

            # Stop if we have enough
            if not fetch_all and len(issues) >= limit:
                issues = issues[:limit]
                break

            page += 1
            _check_rate_limit(response, repo)

    return issues


def write_csv(issues, output_path):
    """Write issues list to a CSV file."""
    fieldnames = [
        "id",
        "number",
        "title",
        "body",
        "state",
        "created_at",
        "updated_at",
        "closed_at",
        "user_login",
        "labels",
        "comments",
        "url",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for issue in tqdm(issues, desc="Writing CSV", unit="issue"):
            writer.writerow(
                {
                    "id": issue.get("id"),
                    "number": issue.get("number"),
                    "title": issue.get("title", ""),
                    "body": issue.get("body") or "",
                    "state": issue.get("state"),
                    "created_at": issue.get("created_at"),
                    "updated_at": issue.get("updated_at"),
                    "closed_at": issue.get("closed_at"),
                    "user_login": (issue.get("user") or {}).get("login", ""),
                    "labels": ",".join(lbl["name"] for lbl in issue.get("labels", [])),
                    "comments": issue.get("comments", 0),
                    "url": issue.get("html_url", ""),
                }
            )


def _get_parser():
    try:
        from tdsuite.cli import get_fetch_issues_parser
        return get_fetch_issues_parser()
    except ImportError:
        import argparse

        p = argparse.ArgumentParser(
            description="Fetch GitHub issues from a public repo into CSV"
        )
        p.add_argument("--repo", required=True, help="owner/repo")
        p.add_argument("--output", default="issues.csv")
        p.add_argument("--state", choices=["open", "closed", "all"], default="all")
        p.add_argument("--token", default=None)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--all", action="store_true", dest="fetch_all")
        g.add_argument("--limit", type=int, default=100, metavar="N")
        return p


def main():
    args = _get_parser().parse_args()

    mode = "all" if args.fetch_all else f"latest {args.limit}"
    print(f"Fetching {mode} '{args.state}' issues from {args.repo}...")

    issues = fetch_issues(
        repo=args.repo,
        state=args.state,
        token=args.token,
        limit=getattr(args, "limit", 100),
        fetch_all=getattr(args, "fetch_all", False),
    )

    print(f"Fetched {len(issues)} issues. Writing to {args.output}...")
    write_csv(issues, args.output)
    print(f"Done. Saved to {args.output}")
    print(
        f"\nNext step — clean the body text:\n"
        f"  python scripts/extract_issue_bodies.py --input {args.output} --output issue_texts.csv"
    )


if __name__ == "__main__":
    main()
