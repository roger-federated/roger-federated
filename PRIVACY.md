# Privacy Notice: Roger Federation Server

**Last updated: 2026-07-02**

This privacy notice applies to the default federation server operated by the roger project. Self-hosted federation servers are operated by their respective administrators, who are controllers for their own deployments; this document does not apply to those systems.

## 1. Controller Identity and Contact

The data controller for the default roger federation server is:

Thijs van Weezel  
Email: contact@rogerfederated.com

No data protection officer is appointed; none is required under Article 37 GDPR given the nature, scale, and context of this processing (no systematic large-scale monitoring, no large-scale special-category data).

## 2. Data Processed

The federation server processes two distinct categories of data:

### Access Log Data
Every HTTP request to the server (including `/status` checks, `/contribute` uploads, and `/pull` downloads) generates a standard web server access log entry, containing:
- Client IP address
- Request timestamp
- HTTP method and request path
- HTTP response status

This data is collected automatically by the reverse proxy and application server, the same way any web service logs requests.

### Protocol Payload Data
When a user calls `/contribute` to upload a gradient update, the following protocol data is transmitted:
- Client protocol version (an integer)
- `model_id`: identifier of the base model the update applies to
- Round identifier
- Masked ΔW share: a cryptographic secret-share of the user's local gradient update, produced via secure multi-party computation (Bonawitz-style X25519 key exchange and SHAKE-based pairwise masking)

### What Is Explicitly Not Collected
- Raw conversation or session content from the user's machine
- Unencrypted model weights, deltas, or unmasked gradient contributions from individual users
- Any account, identity, or authentication data (no user accounts, no email collection, no login system)
- Analytics or telemetry beyond the access-log fields above

## 3. Purpose and Legal Basis

### Access Log Data
**Purpose:** operating, securing, debugging, and maintaining the federation server; detecting and preventing abuse; investigating incidents.  
**Legal Basis (GDPR Article 6(1)(f)):** Legitimate interest — running and securing the service. This basis was assessed under the three-part test described in EDPB Guidelines 1/2024 on Article 6(1)(f) (purpose, necessity, balancing): the interest in operating and securing a network service is genuine and clearly articulated; the fields collected are the technical minimum needed to run an HTTP service and are not used for any other purpose; and, given their limited scope and the retention schedule in Section 6, they do not override your rights and freedoms. You have the right to object to this processing at any time on grounds relating to your particular situation (Section 8, Article 21).

### Protocol Payload Data
**Purpose:** aggregating gradient contributions from multiple users into a shared community model update, which is then made available to all clients.  
**Legal Basis (GDPR Article 6(1)(f)):** Legitimate interest — not consent. Federated gradient contribution is enabled by default and can be turned off in the local configuration (opt-out), rather than requiring an affirmative opt-in. Under Article 4(11) and Article 7 GDPR and Recital 32 ("[s]ilence, pre-ticked boxes or inactivity should not ... constitute consent"), a default-on setting does not meet the bar for valid consent; this reading is reinforced by EDPB Guidelines 05/2020 on consent and the CJEU's Planet49 judgment (C-673/17), both of which treat pre-set defaults requiring an opt-out as invalid. This processing therefore relies on the legitimate interest of the controller and the contributing community in producing a shared model update, necessity being limited to masked, individually non-reconstructable gradient shares (Section 7), balanced by the mitigations in Section 7 (secure aggregation, cohort minimums, client-side PII filtering, differential-privacy cold start) and by your unconditional right to object at any time by disabling federation in your local configuration (Section 8, Article 21) — at least as easy as withdrawing consent would be under Article 7(3).

Providing this data is not a statutory or contractual requirement (Article 13(2)(e)): federated contribution is entirely optional, and roger's other functionality works without it. Access log fields (e.g. your IP address) are an unavoidable technical consequence of making any HTTP request to the server; if you do not want that data processed, do not connect to this server (a self-hosted federation server, operated under its own controller, is an alternative — see the introductory note above).

## 4. Recipients and Subprocessors

The default federation server is hosted on Scaleway, a cloud infrastructure provider operating under EU GDPR. No data is shared with any other recipients or subprocessors. No data is sold or shared with advertisers, analytics services, or any third party outside the Scaleway hosting infrastructure.

## 5. International Data Transfers

The default federation server is hosted in Scaleway's Amsterdam data center (nl-ams), located in the Netherlands within the European Economic Area (EEA). Data does not leave the EEA under the default deployment. Users self-hosting a federation server on their own infrastructure are responsible for their own data transfer compliance.

## 6. Data Retention

**Access log data:** the server does not persist access logs itself; they exist only as the hosting platform's default container log capture (currently Scaleway Cockpit), retained for 7 days before automatic deletion. This is Scaleway's default retention for logs & traces and is proportionate to the operational-security and debugging purpose described in Section 3; it is configurable by the controller in the hosting platform's console.

**Protocol payload data (individual contributions):** an individual masked or DP-noised gradient upload is staged in object storage only for the duration of processing its round, typically seconds to a few minutes, and is deleted immediately once folded into the aggregate (whether the fold succeeds or fails). If a round is interrupted by a crash or scale-to-zero shutdown before it can clean up after itself, an object-storage lifecycle rule independently expires any orphaned staged objects within a few hours. No individual contribution is retained beyond that window; retention of an individual contribution is, in practice, effectively zero.

**The aggregated global model** (the shared community delta produced by folding contributions together) is retained indefinitely, since it is the ongoing product of the service, not personal data of any individual contributor. Once a masked gradient contribution is folded into it, it is designed to be cryptographically unattributable to any individual contributor and cannot be technically separated or extracted (see Section 8).

## 7. Security Measures

The following technical safeguards protect the data processed by the federation server:

- **Secure aggregation:** Individual masked ΔW contributions are cryptographically secret-shared via Bonawitz-style secure aggregation (X25519 key exchange + SHAKE-based pairwise masking). The server cannot decrypt an individual contribution; only the cryptographically combined aggregate of contributions from a cohort is decrypted and processed.
- **Cohort minimums:** A round is aggregated only if at least k_min contributors (typically 3 or higher) have participated. No round is aggregated from a single contributor or a sub-threshold cohort, preventing trivial attribution of updates to individuals.
- **Client-side privacy filter:** Before any local gradient is computed, a client-side privacy filter anonymizes personally identifiable information in the user's session data by swapping it for consistent surrogate values at the token level. This runs entirely on the user's machine, before any network call, reducing the amount of PII that reaches the training process.
- **Cold-start differential privacy:** During the bootstrap phase before sufficient contributors are online for full secure aggregation, updates are processed through a differential-privacy-style noised mechanism rather than unprotected secure aggregation.
- **TLS in transit:** All communication between client and server is encrypted in transit using TLS.
- **Encryption at rest:** The storage backend (S3) encrypts data at rest.

**Data Protection Impact Assessment (Article 35):** this processing was screened against the nine criteria published by the EU data protection authorities (WP248, adopted by the EDPB and referenced in Dutch AP guidance), under which a DPIA is generally required if two or more criteria are met. This processing meets only one: use of an innovative technology (secure multi-party aggregation of ML gradient updates). It does not involve evaluation or scoring of individuals, automated decision-making with legal or similarly significant effect, systematic monitoring, special-category or highly personal data, large-scale matching of datasets, or vulnerable data subjects; the client-side privacy filter, secure aggregation, cohort minimums, and differential-privacy cold-start mechanism were specifically adopted to keep the risk to data subjects low. On that basis, a formal DPIA is not considered required. This processing does not appear to fall within any of the fixed categories of processing for which the Dutch DPA (Autoriteit Persoonsgegevens) mandates a DPIA regardless of the nine-criteria screening (e.g. large-scale biometric or health data processing, credit scoring, public camera surveillance); the controller should periodically re-confirm this against the AP's published list as it may be updated.

## 8. Data Subject Rights

You have the following rights under GDPR:

- **Access (Article 15):** You may request a copy of personal data we hold about you. Due to the architecture of secure aggregation, we do not retain individual unmasked contributions once they are aggregated; if your contribution has been folded into a cumulative aggregate, the individual data cannot be extracted or reconstructed.
- **Rectification (Article 16):** You may request that inaccurate personal data be corrected. Because contributions are designed to be cryptographically unattributable once aggregated, rectification of a folded contribution is not technically possible.
- **Erasure (Article 17):** You may request deletion of personal data. Access logs can be deleted within the retention periods described in Section 6. Masked contributions that have already been aggregated into the cumulative global delta cannot be technically separated or erased due to the security-by-design properties of secure aggregation (the entire point is that the server cannot identify or extract an individual contribution post-aggregation).
- **Restriction of processing (Article 18):** You may request that we restrict the processing of your data; you can disable federation in your local configuration immediately, which will stop any future contributions.
- **Objection (Article 21):** You may object to processing; you can disable federation in your local configuration at any time.
- **Portability (Article 20):** You may request a copy of personal data in a portable format. Access logs and contribution metadata can be provided; masked shares themselves, once aggregated, are not portable as meaningful data due to cryptographic masking.

For Access, Rectification, and Erasure of a post-aggregation contribution specifically, the governing provision is Article 11(2) GDPR ("processing which does not require identification"), not an Article 17(3) exception (none of the Article 17(3) grounds — freedom of expression, legal obligation, public-interest archiving, legal claims — actually fit this case): once secure aggregation has run, the controller is no longer in a position to identify which contributor a given share came from, so Articles 15–20 do not apply to that folded contribution, except where you can supply additional information that would re-enable identification (which the cryptographic design of secure aggregation is intended to prevent). This does not affect these rights over your access log data, which remains identifiable (by IP address) and is handled as described above.

To exercise any of these rights, contact contact@rogerfederated.com.

## 9. Right to Object / Opt Out and Lodge Complaints

Both categories of processing described in Section 3 rely on legitimate interest (Article 6(1)(f)), not consent, so this section describes your right to object (Article 21) rather than a right to withdraw consent. You may object to, and opt out of, federated gradient contribution at any time by disabling federation in your local roger configuration or switching to a self-hosted federation server. Opting out does not affect the lawfulness of processing carried out before you opted out.

You also have the right to lodge a complaint with a supervisory data protection authority, such as the Dutch Data Protection Authority (Autoriteit Persoonsgegevens) if you believe your rights have been violated.

## 10. Automated Decision-Making and Profiling (Article 22)

No automated decision-making or profiling of any person occurs within the roger federation system. Your contributions are aggregated into a shared model update; no individual scoring, behavioral profiling, or automated classification of users is performed.

## 11. Children

The roger CLI tool is not directed at, or intended for use by, individuals under the age of 16. If we become aware that a child under 16 has provided personal data, we will take steps to delete such data promptly.

## 12. Changes to This Policy

Material changes to this privacy notice will be communicated to users via the roger CLI's existing client-version-check mechanism. When you run the tool, it checks the server for available updates and notifies you if your client version is outdated or if a newer version is available; the same notification channel will be used to alert users to material privacy-policy changes. The date at the top of this document ("Last updated") will be updated whenever changes are made.

## 13. Privacy Notice Presentation to Users

Before a user's first contribution to a federation, the roger CLI displays a one-time privacy notice pointing to this document and explaining what is processed, on what basis, and how to opt out (Section 9) before any contribution is made. If this policy changes materially, the CLI will surface the change to the user via the client-version-check notice mechanism described in Section 12, and the one-time notice will reappear on next startup until acknowledged.

---

For questions about this privacy notice or roger's data practices, contact contact@rogerfederated.com.
