# Terraform Notes

Terraform is intentionally left as service scaffolding rather than a fully provisioned cloud account template. A full environment would define:

- Artifact Registry repository
- Cloud Run service
- IAM bindings
- log-based metrics
- alerting policies
- service-account permissions

Keeping this as documentation avoids storing account-specific or organization-specific infrastructure settings in the repo.
