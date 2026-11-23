"""Connected, view-scoped appearance permissions for Reference13 only."""
from .state_13 import part_views


ROLE_PARTS = {
    'face': ('face',),
    'head': ('head_hair', 'face'),
    'clothing': ('upper_clothing', 'back_clothing', 'lower_body'),
    'clothing_front': ('upper_clothing', 'lower_body'),
    'clothing_back': ('upper_clothing', 'back_clothing', 'lower_body'),
    'hands': ('hands',),
    'footwear': ('footwear',),
}
ROLE_NAMES = {'face': 'facial appearance', 'head': 'head appearance',
              'clothing': 'outfit', 'clothing_front': 'front of the outfit',
              'clothing_back': 'back of the outfit', 'hands': 'hands and gloves',
              'footwear': 'footwear'}
REAR_VIEWS = frozenset(('face_back', 'body_back', 'clothing_back'))


def _role_segments(role):
    """A reference slot may carry multiple '+'-joined part roles (same file)."""
    return [seg for seg in role.split('+') if seg]


def build_appearance_policy(state, references):
    """Connected, view-scoped appearance permissions; one entry per role segment."""
    policy = []
    for reference in references:
        for role in _role_segments(reference['role']):
            scopes = {}
            for part in ROLE_PARTS.get(role, ()):
                views = part_views(state, part)
                if role == 'face':
                    views = tuple(view for view in views if view not in REAR_VIEWS)
                elif role == 'clothing_front':
                    views = tuple(view for view in views if view not in REAR_VIEWS)
                elif role == 'clothing_back':
                    # A fixed side camera may reveal a rear garment surface.
                    views = tuple(view for view in views if view in REAR_VIEWS or view == 'body_left')
                if views:
                    scopes[part] = views
            if scopes:
                policy.append({**reference, 'role': role, 'parts': scopes})
    return tuple(policy)


def unused_reference_instructions(references, appearance_policy):
    eligible_sources = {item['source'] for item in appearance_policy}
    messages = {}
    for ref in references:
        if ref['source'] in eligible_sources:
            continue
        unused = [role for role in _role_segments(ref['role'])
                  if role in ROLE_PARTS]
        if not unused:
            continue
        messages[ref['source']] = (
            f"{ref['tag']} has role {'+'.join(unused)}, but no eligible part is visible in the selected views. "
            'Ignore it for appearance changes.'
        )
    return messages


def applicable_references(policy, parts, view=None):
    return tuple(reference for reference in policy
                 if any(part in reference['parts'] and
                        (view is None or view in reference['parts'][part]) for part in parts))


def reference_sources(references):
    return ', '.join(f"{item['tag']} ({item['role']})" for item in references)
