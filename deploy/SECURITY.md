# EC2 Pilot Security Checklist (RAG-4i-Cloud)

Review before and after every deployment. Nothing here is automated on
purpose — a human confirms each item.

## Network

- [ ] Security group opens **only** TCP 22 (operator IP `/32`) and
      TCP 8501 (trusted operator IPs `/32` each). Never `0.0.0.0/0`.
- [ ] Port 1234 (LM Studio) is reachable **only** via the reverse SSH
      tunnel from the operator PC. No SG rule mentions 1234.
- [ ] RDS `rag4i-db` has public access OFF; port 5432 reachable only
      from the EC2 security group (plus any pre-existing operator `/32`
      needed for maintenance — remove when done).
- [ ] S3 bucket keeps Block Public Access ON, SSE-S3 ON, versioning ON.

## Identity & secrets

- [ ] EC2 uses the least-privilege IAM role (read-only on
      `rag4i-company-documents-305740358559-ap-south-2-an/documents/*`).
      No `PutObject`/`DeleteObject` unless a write flow was approved.
- [ ] No AWS access keys anywhere: not in `.env`, not in code, not in git.
- [ ] `DB_PASSWORD` exists **only** in `/home/ec2-user/RAG-4i-Cloud/.env`
      on the instance (never committed; `.env` is git-ignored).
- [ ] SSH key (`rag4i-admin-key.pem`) has owner-only permissions and is
      never committed or pasted into chat.

## Application assumptions (do not rely on being changed silently)

- [ ] The app binds `0.0.0.0:8501`; access control lives **only** in the SG.
- [ ] The reverse tunnel (`-R 1234:127.0.0.1:1234`) must be re-established
      after every reboot/logout of the operator PC.
- [ ] EC2 public IP changes on stop/start — update SG rules, tunnel
      target, and the browser URL together.
- [ ] Pilot serves test/non-sensitive documents only (no HTTPS/auth yet).
