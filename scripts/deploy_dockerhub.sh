#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/deploy_dockerhub.sh --username DOCKERHUB_USERNAME [options]

Options:
  -u, --username USERNAME   Docker Hub username or organization. Required.
  -i, --image-name NAME     Docker Hub repository name. Default: aws-documentation-rag
  -t, --tag TAG             Image tag. Default: latest
  -f, --dockerfile PATH     Dockerfile path. Default: Dockerfile
  -c, --context PATH        Build context path. Default: .
  -y, --yes                 Skip the interactive confirmation prompt.
  -h, --help                Show this help text.

Example:
  scripts/deploy_dockerhub.sh --username mydockerhubuser --tag v1
USAGE
}

dockerhub_username=""
image_name="aws-documentation-rag"
tag="latest"
dockerfile="Dockerfile"
context="."
assume_yes="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--username)
      dockerhub_username="${2:-}"
      shift 2
      ;;
    -i|--image-name)
      image_name="${2:-}"
      shift 2
      ;;
    -t|--tag)
      tag="${2:-}"
      shift 2
      ;;
    -f|--dockerfile)
      dockerfile="${2:-}"
      shift 2
      ;;
    -c|--context)
      context="${2:-}"
      shift 2
      ;;
    -y|--yes)
      assume_yes="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$dockerhub_username" ]]; then
  echo "Docker Hub username is required." >&2
  usage >&2
  exit 2
fi

if [[ -z "$image_name" || -z "$tag" ]]; then
  echo "Image name and tag must not be empty." >&2
  exit 2
fi

if [[ ! -f "$dockerfile" ]]; then
  echo "Dockerfile not found: $dockerfile" >&2
  exit 1
fi

if [[ ! -d "$context" ]]; then
  echo "Build context directory not found: $context" >&2
  exit 1
fi

full_image_name="${dockerhub_username}/${image_name}:${tag}"

echo "Docker image to build and push:"
echo "  ${full_image_name}"
echo

if [[ "$assume_yes" != "true" ]]; then
  read -r -p "Confirm this image tag before deployment [y/N]: " confirm
  case "$confirm" in
    y|Y|yes|YES)
      ;;
    *)
      echo "Deployment cancelled."
      exit 0
      ;;
  esac
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker does not appear to be running or accessible." >&2
  exit 1
fi

echo "Building ${full_image_name}..."
docker build \
  --file "$dockerfile" \
  --tag "$full_image_name" \
  "$context"

echo "Pushing ${full_image_name}..."
docker push "$full_image_name"

echo "Deployment complete: ${full_image_name}"
